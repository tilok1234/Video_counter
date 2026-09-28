package com.palletcounter.core.counting

import com.palletcounter.core.geometry.Box
import com.palletcounter.core.tracking.MotionEstimate
import com.palletcounter.core.tracking.Track
import com.palletcounter.core.tracking.TrackState
import com.palletcounter.core.tracking.TrackerOutput
import kotlinx.serialization.Serializable
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sign

@Serializable
data class CounterConfig(
    /** Vertical count line position (normalized x of the upright frame). */
    val lineX: Float = 0.5f,
    /**
     * Half-width of the dead band around the line. A crossing is only registered when a
     * track is observed outside the band on one side and later outside it on the other,
     * so jitter of a pallet parked on the line never changes the count.
     */
    val hysteresis: Float = 0.04f,
    /** Matched frames before a track may change the count. */
    val minHits: Int = 3,
    /** A track whose best detection score stays below this never counts. */
    val minTrackConfidence: Float = 0.5f,
    /** Crossings observed before a track became eligible are applied later if this recent. */
    val pendingCrossingMaxAgeMillis: Long = 1500,
    /**
     * When a new track appears where a recently lost/removed track is predicted to be, it
     * inherits that track's line state (short-term memory against duplicate counts).
     */
    val stitchEnabled: Boolean = true,
    val stitchWindowMillis: Long = 1500,
    /** Minimum DIoU between the motion-predicted old box and the new track. */
    val stitchMinDiou: Float = 0.0f,
    val stitchMaxSizeRatio: Float = 1.5f,
    /**
     * Ignore tracks whose median box height is below this fraction of the typical height of
     * counted pallets (e.g. far-row pallets visible through gaps). 0 disables the filter.
     */
    val relativeSizeFilter: Float = 0f,
    /** Confirmed tracks needed before the size filter becomes active. */
    val sizeFilterMinReference: Int = 3,
    /** Warn when pallets move more than this fraction of their width between detector frames. */
    val tooFastDisplacementRatio: Float = 0.35f,
    /** Camera speeds below this (frame widths / s) are reported as stationary. */
    val stationarySpeed: Float = 0.02f,
)

@Serializable
enum class LineSide { LEFT, RIGHT }

/**
 * One applied line crossing. [direction] is +1 for a pallet moving left→right across the
 * line and -1 for right→left.
 */
@Serializable
data class CountEvent(
    val trackId: Int,
    val timestampNanos: Long,
    val direction: Int,
    val netCountAfter: Int,
    val box: Box,
    /** Id of the lost/removed track this one was stitched to, if any. */
    val stitchedFrom: Int? = null,
    /** How long the crossing waited for the track to become eligible (confirmation). */
    val delayedMillis: Long = 0,
)

/** Per-track counting state exposed for overlays and debugging. */
@Serializable
data class TrackCountInfo(
    val trackId: Int,
    val side: LineSide?,
    /** Net contribution of this track to the signed count: -1, 0 or +1. */
    val contribution: Int,
    val eligible: Boolean,
    val pendingCrossings: Int,
    val sizeFiltered: Boolean,
    val stitchedFrom: Int?,
    val crossings: Int,
)

@Serializable
data class MotionStatus(
    /** Camera-induced image motion in frame widths / heights per second. */
    val vx: Float = 0f,
    val vy: Float = 0f,
    val valid: Boolean = false,
    /** Pallet displacement between consecutive detector frames relative to pallet width. */
    val displacementRatio: Float = 0f,
    val tooFast: Boolean = false,
    val stationary: Boolean = false,
    /** Sign of the image motion of pallets: +1 moving right, -1 moving left, 0 unknown. */
    val direction: Int = 0,
)

class CounterUpdate(
    /** Signed net crossings: left→right minus right→left. */
    val netCount: Int,
    val events: List<CountEvent>,
    val trackInfo: Map<Int, TrackCountInfo>,
    val motion: MotionStatus,
    /** Tracks the tracker should drop because they were merged into a new track. */
    val tracksToRemove: List<Int>,
    val stats: CounterStats,
)

@Serializable
data class CounterStats(
    val leftToRight: Int = 0,
    val rightToLeft: Int = 0,
    val stitches: Int = 0,
    val delayedCrossings: Int = 0,
    val discardedPendingCrossings: Int = 0,
    val sizeFilteredTracks: Int = 0,
    /** Confirmed tracks that were first seen past the count line and never crossed it. */
    val lateTracks: Int = 0,
)

/**
 * Counts pallets as signed crossings of a vertical line.
 *
 * Every crossing adds its direction (+1 left→right, -1 right→left) to a signed total and
 * the displayed count is its absolute value. Consequences:
 *
 * - a pallet visible for many frames crosses once and is counted once;
 * - a pallet jittering on the line changes nothing (hysteresis band);
 * - walking back over pallets un-counts them, re-walking counts them again, so a section
 *   accidentally scanned twice is not double counted — even if track ids change;
 * - the walking direction never has to be detected or configured.
 *
 * Crossings of tentative (unconfirmed) tracks are held and applied once the track is
 * confirmed, or dropped if it dies first (most false positives).
 */
class LineCounter(config: CounterConfig = CounterConfig()) {
    var config: CounterConfig = config
        private set

    private class State(val trackId: Int) {
        /** Last side observed outside the hysteresis band. */
        var side: LineSide? = null
        /** First side ever observed (inherited through stitching). */
        var firstSide: LineSide? = null
        var contribution = 0
        var crossings = 0
        var everEligible = false
        var eligible = false
        var sizeFiltered = false
        var stitchedFrom: Int? = null
        /** Provisional link to a LOST track this one probably continues. */
        var linkedTo: Int? = null
        /** Ghost to merge in once this new track's first side is known. */
        var ghostToMerge: State? = null
        var referenceRecorded = false
        val pending = ArrayList<Pair<Int, Long>>() // (direction, timestamp)
        val heights = ArrayDeque<Float>()

        fun medianHeight(): Float {
            if (heights.isEmpty()) return 0f
            val s = heights.sorted()
            return s[s.size / 2]
        }
    }

    /** A removed track's state, with its last predicted box and the time of that prediction. */
    private class Ghost(val state: State, val box: Box, val anchorNanos: Long)

    private val states = HashMap<Int, State>()
    private val ghosts = ArrayList<Ghost>()

    /** Median heights of recent confirmed tracks; the upper quartile is the "near row" size. */
    private val referenceHeights = ArrayDeque<Float>()
    private var netCount = 0
    private var stats = CounterStats()
    private var lastTimestamp: Long? = null

    val signedCount: Int get() = netCount
    val count: Int get() = abs(netCount)

    fun reset() {
        states.clear()
        ghosts.clear()
        referenceHeights.clear()
        netCount = 0
        stats = CounterStats()
        lastTimestamp = null
    }

    /**
     * Changes line/filter parameters mid-scan. If the line moved, every track's side is
     * re-seeded so that moving the line can never create crossings by itself.
     */
    fun updateConfig(newConfig: CounterConfig) {
        val lineMoved = newConfig.lineX != config.lineX || newConfig.hysteresis != config.hysteresis
        config = newConfig
        if (lineMoved) {
            for (s in states.values) {
                s.side = null
                s.firstSide = null
                s.pending.clear()
                s.linkedTo = null
                s.ghostToMerge = null
            }
            ghosts.clear()
        }
    }

    /**
     * Processes one tracker update in five phases:
     * 1. removed tracks are merged into a linked successor or kept as ghosts;
     * 2. new tracks are linked to a matching lost track or ghost;
     * 3. sides relative to the line are observed, crossings become pending;
     * 4. links are resolved (merge once the successor is confirmed, drop if the old track
     *    was recovered);
     * 5. pending crossings of eligible tracks are applied to the count.
     */
    fun update(output: TrackerOutput): CounterUpdate {
        val now = output.timestampNanos
        val dt = lastTimestamp?.let { (now - it) / 1e9 } ?: 0.0
        lastTimestamp = now
        val events = ArrayList<CountEvent>()
        val toRemove = ArrayList<Int>()
        val tracksById = output.tracks.associateBy { it.id }

        // 1) Removed tracks.
        for (t in output.removed) {
            val s = states.remove(t.id) ?: continue
            val successor = states.values.firstOrNull { it.linkedTo == t.id }
            if (successor != null) {
                merge(s, successor, now)
                continue
            }
            if (s.everEligible && s.crossings == 0 && isLate(s)) stats = stats.copy(lateTracks = stats.lateTracks + 1)
            // t.box is already predicted to `now`, so the ghost is anchored at `now` too.
            ghosts += Ghost(s, t.box, now)
        }
        val windowNanos = config.stitchWindowMillis * 1_000_000
        ghosts.removeAll { now - it.anchorNanos > windowNanos }

        // 2) New tracks.
        for (t in output.born) {
            val s = State(t.id)
            states[t.id] = s
            if (!config.stitchEnabled) continue
            when (val target = findStitchTarget(t, output, now)) {
                is StitchTarget.Lost -> s.linkedTo = target.trackId
                is StitchTarget.Removed -> s.ghostToMerge = target.state
                null -> Unit
            }
        }

        // 3) Observe sides.
        for (t in output.tracks) {
            val s = states.getOrPut(t.id) { State(t.id) }
            if (!t.updatedThisFrame) continue
            s.heights.addLast(t.lastDetection.box.height)
            if (s.heights.size > HEIGHT_SAMPLES) s.heights.removeFirst()
            if (!s.referenceRecorded && t.state == TrackState.CONFIRMED && s.heights.size >= config.minHits) {
                s.referenceRecorded = true
                referenceHeights.addLast(s.medianHeight())
                if (referenceHeights.size > REFERENCE_SAMPLES) referenceHeights.removeFirst()
            }
            s.sizeFiltered = isSizeFiltered(s)
            s.eligible = t.state == TrackState.CONFIRMED &&
                t.hits >= config.minHits &&
                t.maxConfidence >= config.minTrackConfidence &&
                !s.sizeFiltered
            if (s.eligible) s.everEligible = true

            val x = t.box.centerX
            val side = when {
                x < config.lineX - config.hysteresis -> LineSide.LEFT
                x > config.lineX + config.hysteresis -> LineSide.RIGHT
                else -> null
            }
            if (side != null) {
                val previous = s.side
                if (s.firstSide == null) s.firstSide = side
                s.side = side
                if (previous != null && previous != side) s.pending += direction(previous, side) to now
            }
            s.ghostToMerge?.let { ghost ->
                s.ghostToMerge = null
                merge(ghost, s, now)
            }
        }

        // 4) Resolve provisional links to lost tracks.
        for (s in states.values) {
            val oldId = s.linkedTo ?: continue
            val old = tracksById[oldId]
            val oldState = states[oldId]
            val me = tracksById[s.trackId]
            when {
                old == null || oldState == null -> s.linkedTo = null
                old.state != TrackState.LOST -> s.linkedTo = null // the old track was recovered
                me != null && me.state == TrackState.CONFIRMED -> {
                    states.remove(oldId)
                    toRemove += oldId
                    merge(oldState, s, now)
                }
            }
        }

        // 5) Apply pending crossings.
        for (t in output.tracks) {
            if (t.id in toRemove) continue
            val s = states[t.id] ?: continue
            if (s.eligible) {
                for ((direction, at) in s.pending) {
                    netCount += direction
                    s.contribution += direction
                    s.crossings++
                    val delayed = (now - at) / 1_000_000
                    stats = if (direction > 0) stats.copy(leftToRight = stats.leftToRight + 1)
                    else stats.copy(rightToLeft = stats.rightToLeft + 1)
                    if (delayed > 0) stats = stats.copy(delayedCrossings = stats.delayedCrossings + 1)
                    events += CountEvent(t.id, now, direction, netCount, t.box, s.stitchedFrom, delayed)
                }
                s.pending.clear()
            } else {
                val maxAge = config.pendingCrossingMaxAgeMillis * 1_000_000
                val before = s.pending.size
                s.pending.removeAll { now - it.second > maxAge }
                val dropped = before - s.pending.size
                if (dropped > 0) {
                    stats = stats.copy(discardedPendingCrossings = stats.discardedPendingCrossings + dropped)
                }
            }
        }
        // States of tracks that vanished without a "removed" notification (defensive).
        states.keys.retainAll(tracksById.keys)

        stats = stats.copy(sizeFilteredTracks = states.values.count { it.sizeFiltered })
        val info = states.values.associate { s ->
            s.trackId to TrackCountInfo(
                trackId = s.trackId,
                side = s.side,
                contribution = s.contribution,
                eligible = s.eligible,
                pendingCrossings = s.pending.size,
                sizeFiltered = s.sizeFiltered,
                stitchedFrom = s.stitchedFrom ?: s.linkedTo,
                crossings = s.crossings,
            )
        }
        return CounterUpdate(
            netCount = netCount,
            events = events,
            trackInfo = info,
            motion = motionStatus(output.motion, dt, tracksById.values),
            tracksToRemove = toRemove,
            stats = stats,
        )
    }

    /**
     * Makes [into] the continuation of [from]: the combined trajectory keeps [from]'s line
     * history, and if the object changed side while neither track observed it, that crossing
     * is added as pending.
     */
    private fun merge(from: State, into: State, now: Long) {
        val gapCrossing = from.side?.let { a ->
            into.firstSide?.takeIf { it != a }?.let { b -> direction(a, b) }
        }
        if (into.side == null) into.side = from.side
        into.firstSide = from.firstSide ?: into.firstSide
        into.contribution += from.contribution
        into.crossings += from.crossings
        into.everEligible = into.everEligible || from.everEligible
        val merged = ArrayList<Pair<Int, Long>>(from.pending)
        if (gapCrossing != null) merged += gapCrossing to now
        merged += into.pending
        into.pending.clear()
        into.pending += merged
        val heights = (from.heights + into.heights).takeLast(HEIGHT_SAMPLES)
        into.heights.clear()
        into.heights += heights
        into.stitchedFrom = from.trackId
        into.linkedTo = null
        stats = stats.copy(stitches = stats.stitches + 1)
    }

    private fun direction(from: LineSide, to: LineSide): Int = if (to == LineSide.RIGHT && from == LineSide.LEFT) 1 else -1

    private sealed interface StitchTarget {
        data class Lost(val trackId: Int) : StitchTarget
        class Removed(val state: State) : StitchTarget
    }

    private fun findStitchTarget(newTrack: Track, output: TrackerOutput, now: Long): StitchTarget? {
        val nb = newTrack.box
        var bestScore = Float.NEGATIVE_INFINITY
        var best: StitchTarget? = null
        var bestGhost: Ghost? = null
        val windowNanos = config.stitchWindowMillis * 1_000_000
        val alreadyLinked = states.values.mapNotNull { it.linkedTo }.toSet()

        // Live LOST tracks: the tracker already predicts them with camera-motion compensation.
        for (t in output.tracks) {
            if (t.id == newTrack.id || t.state != TrackState.LOST || t.id in alreadyLinked) continue
            if (now - t.lastUpdateNanos > windowNanos || t.id !in states) continue
            if (!sizeCompatible(t.box, nb)) continue
            val score = t.box.diou(nb)
            if (score >= config.stitchMinDiou && score > bestScore) {
                bestScore = score
                best = StitchTarget.Lost(t.id)
                bestGhost = null
            }
        }
        // Removed tracks: extrapolate with the current camera-motion estimate.
        val m = output.motion
        for (g in ghosts) {
            val dt = (now - g.anchorNanos) / 1e9f
            val predicted = if (m.valid) g.box.translated(m.vx * dt, m.vy * dt) else g.box
            if (!sizeCompatible(predicted, nb)) continue
            val score = predicted.diou(nb)
            if (score >= config.stitchMinDiou && score > bestScore) {
                bestScore = score
                best = StitchTarget.Removed(g.state)
                bestGhost = g
            }
        }
        bestGhost?.let { ghosts.remove(it) }
        return best
    }

    private fun sizeCompatible(a: Box, b: Box): Boolean {
        val ratio = max(a.height, b.height) / max(1e-4f, min(a.height, b.height))
        return ratio <= config.stitchMaxSizeRatio
    }

    private fun isSizeFiltered(s: State): Boolean {
        val ratio = config.relativeSizeFilter
        if (ratio <= 0f || referenceHeights.size < config.sizeFilterMinReference) return false
        val sorted = referenceHeights.sorted()
        val reference = sorted[(sorted.size * 3) / 4]
        return s.medianHeight() < ratio * reference
    }

    /** True if the track was first seen on the side pallets are moving towards. */
    private fun isLate(s: State): Boolean {
        val dir = netCount.sign
        if (dir == 0) return false
        val passedSide = if (dir > 0) LineSide.RIGHT else LineSide.LEFT
        return s.firstSide == passedSide
    }

    private fun motionStatus(m: MotionEstimate, dt: Double, tracks: Collection<Track>): MotionStatus {
        if (!m.valid) return MotionStatus(direction = netCount.sign)
        val widths = tracks.filter { it.state == TrackState.CONFIRMED && it.updatedThisFrame }.map { it.box.width }
        val typicalWidth = if (widths.isEmpty()) DEFAULT_BOX_WIDTH else widths.sorted()[widths.size / 2]
        val displacement = if (dt > 0) abs(m.vx) * dt.toFloat() / max(typicalWidth, 0.02f) else 0f
        return MotionStatus(
            vx = m.vx,
            vy = m.vy,
            valid = true,
            displacementRatio = displacement,
            tooFast = displacement > config.tooFastDisplacementRatio,
            stationary = abs(m.vx) < config.stationarySpeed,
            direction = if (abs(m.vx) < config.stationarySpeed) 0 else m.vx.sign.toInt(),
        )
    }

    private companion object {
        const val HEIGHT_SAMPLES = 15
        const val REFERENCE_SAMPLES = 25
        const val DEFAULT_BOX_WIDTH = 0.3f
    }
}
