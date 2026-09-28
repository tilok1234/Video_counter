package com.palletcounter.core.tracking

import com.palletcounter.core.detection.Detection
import com.palletcounter.core.detection.DetectionFrame
import kotlinx.serialization.Serializable
import kotlin.math.max
import kotlin.math.min

/**
 * Tracker parameters. Defaults are tuned for 5–15 detector frames per second while walking
 * along a pallet line; all time-based values are in milliseconds so they do not depend on
 * the frame rate.
 */
@Serializable
data class TrackerConfig(
    /** Detections at or above this score take part in the first association round. */
    val highThreshold: Float = 0.5f,
    /** Detections below this are ignored entirely. Between low and high they only extend tracks. */
    val lowThreshold: Float = 0.1f,
    /** Minimum score for an unmatched detection to start a new track. */
    val newTrackThreshold: Float = 0.55f,
    /** Minimum IoU between prediction and detection in the first round. */
    val firstMatchMinIou: Float = 0.2f,
    /** Minimum DIoU for re-associating a LOST track (looser, as predictions drift). */
    val lostMatchMinDiou: Float = -0.1f,
    /** Minimum IoU when extending a confirmed track with a low-score detection. */
    val secondMatchMinIou: Float = 0.4f,
    /** Minimum IoU for matching tentative tracks. */
    val tentativeMatchMinIou: Float = 0.3f,
    /** Height ratio limit when re-associating lost tracks (guards against far-row boxes). */
    val maxSizeRatioForLostMatch: Float = 1.6f,
    /** Weight association similarity by detection score (ByteTrack "fuse score"). */
    val fuseScore: Boolean = true,
    /** Matched frames needed before a track is CONFIRMED. */
    val minHitsToConfirm: Int = 3,
    /** Tentative tracks survive this many consecutive missed frames. */
    val tentativeMaxMissedFrames: Int = 1,
    /** Lost tracks are kept (and predicted) this long before removal. */
    val lostTimeoutMillis: Long = 1500,
    /** Suppress births that overlap an already-updated track by more than this IoU. */
    val duplicateBirthIou: Float = 0.7f,
    /** Measurement noise std as a fraction of box size. */
    val measurementNoise: Float = 0.05f,
    /** Acceleration noise std in box sizes per second squared. */
    val accelerationNoise: Float = 2.0f,
    val sizeAccelerationNoise: Float = 0.5f,
    /** Initial velocity std (box sizes / s) when the camera motion is known / unknown. */
    val velocityPriorStd: Float = 0.3f,
    val unknownVelocityStd: Float = 1.5f,
    /** Drive LOST tracks with the camera-motion estimate instead of their stale velocity. */
    val motionCompensateLostTracks: Boolean = true,
    /** Keep the last camera-motion estimate this long when no track is visible. */
    val motionHoldMillis: Long = 700,
    /** Smoothing factor for the camera-motion estimate (1 = no smoothing). */
    val motionSmoothing: Float = 0.5f,
)

/**
 * Apparent camera motion, estimated as the median velocity of confirmed tracks updated in
 * this frame (all pallets in a line move together in the image while walking past them).
 * Units: normalized frame widths/heights per second.
 */
@Serializable
data class MotionEstimate(
    val vx: Float = 0f,
    val vy: Float = 0f,
    /** Number of tracks that contributed this frame (0 = held or unknown). */
    val samples: Int = 0,
    /** False when nothing has been observed yet or the held estimate expired. */
    val valid: Boolean = false,
)

class TrackerOutput(
    val timestampNanos: Long,
    val dtSeconds: Double,
    /** All tracks that are not removed, including LOST ones. */
    val tracks: List<Track>,
    val born: List<Track>,
    val removed: List<Track>,
    val motion: MotionEstimate,
    /** Detections that were neither matched nor started a track (low-score leftovers). */
    val unmatchedDetections: List<Detection>,
)

/**
 * ByteTrack-style multi-object tracker (Zhang et al., 2022) adapted for variable frame
 * timing and camera-motion compensation:
 *
 * 1. predict every track to the frame timestamp (Kalman, real dt);
 * 2. match high-score detections to confirmed + lost tracks (IoU, DIoU for lost);
 * 3. match low-score detections to still-unmatched confirmed tracks (keeps tracks alive
 *    through motion blur and partial occlusion);
 * 4. match remaining high-score detections to tentative tracks;
 * 5. start tracks from leftovers; age out lost tracks after [TrackerConfig.lostTimeoutMillis];
 * 6. estimate camera motion from the confirmed tracks and use it to drive lost tracks and
 *    to initialise the velocity of new tracks.
 */
class ByteTracker(val config: TrackerConfig = TrackerConfig()) {
    private val tracks = ArrayList<Track>()
    private var nextId = 1
    private var lastTimestamp: Long? = null
    private var motion = MotionEstimate()
    private var lastMotionObservationNanos = Long.MIN_VALUE
    private val externallyRemoved = ArrayList<Track>()

    val activeTracks: List<Track> get() = tracks.toList()

    fun reset() {
        tracks.clear()
        nextId = 1
        lastTimestamp = null
        motion = MotionEstimate()
        lastMotionObservationNanos = Long.MIN_VALUE
        externallyRemoved.clear()
    }

    /** Removes a track immediately (used when the counter merges it into another track). */
    fun removeTrack(id: Int) {
        val t = tracks.firstOrNull { it.id == id } ?: return
        t.state = TrackState.REMOVED
        tracks.remove(t)
        externallyRemoved += t
    }

    fun update(frame: DetectionFrame): TrackerOutput {
        val now = frame.timestampNanos
        val previous = lastTimestamp
        val dt = if (previous == null) 0.0 else ((now - previous) / 1e9).coerceIn(0.0, MAX_DT_SECONDS)
        lastTimestamp = now

        for (t in tracks) {
            t.predict(dt)
            t.updatedThisFrame = false
        }

        val detections = frame.detections.filter { it.confidence >= config.lowThreshold && it.box.isValid }
        val high = detections.filter { it.confidence >= config.highThreshold }
        val low = detections.filter { it.confidence < config.highThreshold }

        val matchedTracks = HashSet<Track>()
        val usedHigh = BooleanArray(high.size)
        val usedLow = BooleanArray(low.size)

        // 1) High-score detections vs confirmed + lost tracks.
        val pool = tracks.filter { it.state == TrackState.CONFIRMED || it.state == TrackState.LOST }
        for ((ti, di) in associate(pool, high, Stage.FIRST)) {
            pool[ti].update(high[di], now, MatchStage.HIGH)
            pool[ti].state = TrackState.CONFIRMED
            matchedTracks += pool[ti]
            usedHigh[di] = true
        }

        // 2) Low-score detections vs remaining confirmed (not lost) tracks.
        val remainingConfirmed = pool.filter { it !in matchedTracks && it.state == TrackState.CONFIRMED }
        for ((ti, di) in associate(remainingConfirmed, low, Stage.SECOND)) {
            remainingConfirmed[ti].update(low[di], now, MatchStage.LOW)
            matchedTracks += remainingConfirmed[ti]
            usedLow[di] = true
        }

        for (t in pool) {
            if (t in matchedTracks) continue
            if (t.state == TrackState.CONFIRMED) t.state = TrackState.LOST
            t.markMissed()
        }

        // 3) Tentative tracks vs remaining high-score detections.
        val tentative = tracks.filter { it.state == TrackState.TENTATIVE }
        val highLeft = high.indices.filter { !usedHigh[it] }
        val highLeftDets = highLeft.map { high[it] }
        for ((ti, dj) in associate(tentative, highLeftDets, Stage.TENTATIVE)) {
            tentative[ti].update(highLeftDets[dj], now, MatchStage.TENTATIVE)
            matchedTracks += tentative[ti]
            usedHigh[highLeft[dj]] = true
        }
        val removed = ArrayList<Track>(externallyRemoved)
        externallyRemoved.clear()
        for (t in tentative) {
            if (t in matchedTracks) {
                if (t.hits >= config.minHitsToConfirm) t.state = TrackState.CONFIRMED
                continue
            }
            t.markMissed()
            if (t.missedFrames > config.tentativeMaxMissedFrames) {
                t.state = TrackState.REMOVED
                removed += t
            }
        }

        // 4) Births from remaining high-score detections.
        val born = ArrayList<Track>()
        val updatedBoxes = tracks.filter { it.updatedThisFrame }.map { it.box }
        for (i in high.indices) {
            if (usedHigh[i]) continue
            val det = high[i]
            if (det.confidence < config.newTrackThreshold) continue
            if (updatedBoxes.any { it.iou(det.box) > config.duplicateBirthIou }) continue
            if (born.any { it.box.iou(det.box) > config.duplicateBirthIou }) continue
            val m = currentMotion(now)
            val track = Track(
                id = nextId++,
                detection = det,
                startNanos = now,
                velocityX = if (m.valid) m.vx else 0f,
                velocityY = if (m.valid) m.vy else 0f,
                hasVelocityPrior = m.valid,
                config = config,
            )
            if (config.minHitsToConfirm <= 1) track.state = TrackState.CONFIRMED
            born += track
        }
        tracks += born

        // 5) Age out lost tracks.
        val timeoutNanos = config.lostTimeoutMillis * 1_000_000
        val it = tracks.iterator()
        while (it.hasNext()) {
            val t = it.next()
            if (t.state == TrackState.REMOVED) {
                it.remove()
            } else if (t.state == TrackState.LOST && now - t.lastUpdateNanos > timeoutNanos) {
                t.state = TrackState.REMOVED
                removed += t
                it.remove()
            }
        }

        // 6) Camera-motion estimate and compensation.
        updateMotion(now)
        if (config.motionCompensateLostTracks && motion.valid) {
            for (t in tracks) if (t.state == TrackState.LOST) t.compensateMotion(motion.vx, motion.vy)
        }

        val bornFrom = born.map { it.lastDetection }.toSet()
        val unmatched = ArrayList<Detection>()
        for (i in high.indices) if (!usedHigh[i] && high[i] !in bornFrom) unmatched += high[i]
        for (i in low.indices) if (!usedLow[i]) unmatched += low[i]

        return TrackerOutput(
            timestampNanos = now,
            dtSeconds = dt,
            tracks = tracks.toList(),
            born = born,
            removed = removed,
            motion = motion,
            unmatchedDetections = unmatched,
        )
    }

    private enum class Stage { FIRST, SECOND, TENTATIVE }

    private fun associate(candidates: List<Track>, dets: List<Detection>, stage: Stage): List<Pair<Int, Int>> {
        if (candidates.isEmpty() || dets.isEmpty()) return emptyList()
        val cost = Array(candidates.size) { FloatArray(dets.size) }
        for (i in candidates.indices) {
            val t = candidates[i]
            val tb = t.box
            for (j in dets.indices) {
                val d = dets[j]
                val similarity: Float
                val feasible: Boolean
                if (stage == Stage.FIRST && t.state == TrackState.LOST) {
                    similarity = tb.diou(d.box)
                    val ratio = max(tb.height, d.box.height) / max(1e-4f, min(tb.height, d.box.height))
                    feasible = similarity >= config.lostMatchMinDiou && ratio <= config.maxSizeRatioForLostMatch
                } else {
                    similarity = tb.iou(d.box)
                    val minIou = when (stage) {
                        Stage.FIRST -> config.firstMatchMinIou
                        Stage.SECOND -> config.secondMatchMinIou
                        Stage.TENTATIVE -> config.tentativeMatchMinIou
                    }
                    feasible = similarity >= minIou
                }
                cost[i][j] = if (!feasible) {
                    Hungarian.INFEASIBLE
                } else {
                    val s = if (config.fuseScore) similarity * d.confidence else similarity
                    1f - s
                }
            }
        }
        return Hungarian.solve(cost)
    }

    private fun updateMotion(now: Long) {
        val contributors = tracks.filter {
            it.updatedThisFrame && it.state == TrackState.CONFIRMED && it.hits >= config.minHitsToConfirm
        }
        if (contributors.isNotEmpty()) {
            val vx = median(contributors.map { it.velocityX })
            val vy = median(contributors.map { it.velocityY })
            val a = config.motionSmoothing
            motion = if (motion.valid) {
                MotionEstimate(motion.vx + a * (vx - motion.vx), motion.vy + a * (vy - motion.vy), contributors.size, true)
            } else {
                MotionEstimate(vx, vy, contributors.size, true)
            }
            lastMotionObservationNanos = now
        } else {
            motion = currentMotion(now).copy(samples = 0)
        }
    }

    private fun currentMotion(now: Long): MotionEstimate {
        if (!motion.valid) return motion
        val held = now - lastMotionObservationNanos
        return if (held > config.motionHoldMillis * 1_000_000) MotionEstimate() else motion
    }

    private companion object {
        /** Larger gaps (app paused, video seek) are clamped so predictions stay sane. */
        const val MAX_DT_SECONDS = 1.0

        fun median(values: List<Float>): Float {
            val s = values.sorted()
            val n = s.size
            return if (n % 2 == 1) s[n / 2] else (s[n / 2 - 1] + s[n / 2]) / 2f
        }
    }
}
