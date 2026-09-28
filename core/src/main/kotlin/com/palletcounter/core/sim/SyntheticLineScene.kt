package com.palletcounter.core.sim

import com.palletcounter.core.detection.Detection
import com.palletcounter.core.detection.DetectionFrame
import com.palletcounter.core.geometry.Box
import kotlinx.serialization.Serializable
import kotlin.math.PI
import kotlin.math.floor
import kotlin.math.sin
import kotlin.random.Random

/** Camera moves along the line at [speedMps] for [durationS] seconds (negative = walking back). */
@Serializable
data class Segment(val durationS: Double, val speedMps: Double)

/** A period with no detections at all (heavy motion blur, a person walking past, …). */
@Serializable
data class Blackout(val startS: Double, val endS: Double)

@Serializable
data class SimConfig(
    val palletCount: Int = 16,
    /** Distance between neighbouring pallet centres along the line (m). */
    val palletPitchM: Double = 1.25,
    /** Length of the pallet face seen by the camera (m). */
    val palletLengthM: Double = 1.2,
    /** First pallet centre relative to the camera start position (m). */
    val firstPalletM: Double = 2.5,
    /** Scene width covered by the image at the distance of the line (m). */
    val viewWidthM: Double = 3.0,
    /** -1: walking forward moves pallets right→left in the image; +1: left→right. */
    val imageDirection: Int = -1,
    val fps: Double = 10.0,
    val boxCenterY: Float = 0.68f,
    val boxHeight: Float = 0.11f,
    /** Std of box-edge jitter (normalized units). */
    val jitter: Float = 0.006f,
    /** Vertical bobbing from walking. */
    val bobAmplitude: Float = 0.01f,
    val bobHz: Double = 1.8,
    val missProbability: Double = 0.08,
    val lowConfidenceProbability: Double = 0.1,
    val falsePositivesPerFrame: Double = 0.05,
    /** Pallets less visible than this fraction at the frame edge are not detected. */
    val minVisibleFraction: Float = 0.35f,
    val blackouts: List<Blackout> = emptyList(),
    /** Emit small, persistent far-row pallet boxes visible through gaps between stacks. */
    val farRowSlivers: Boolean = false,
    val seed: Long = 1,
    /** Infinite line (for the app's live simulation): pallets repeat forever. */
    val endless: Boolean = false,
)

/**
 * Synthetic "walk along a pallet line" generator.
 *
 * This is NOT computer vision. It produces the detections a decent-but-imperfect detector
 * would output for a scripted camera trajectory, so tracking and counting can be tested
 * under controlled failure modes (misses, blackouts, stops, reversals, jerks, false
 * positives, far-row slivers). The Android app uses it only for its clearly labelled
 * SIMULATION mode that tests camera→tracking→counting→UI plumbing without a model.
 */
class SyntheticLineScene(
    val config: SimConfig = SimConfig(),
    val trajectory: List<Segment> = defaultTrajectory(config),
) {
    val durationS: Double = trajectory.sumOf { it.durationS }

    /** Camera position along the line (m) at time [t] seconds. */
    fun cameraPosition(t: Double): Double {
        var pos = 0.0
        var remaining = t
        for (seg in trajectory) {
            if (remaining <= 0) break
            val d = minOf(remaining, seg.durationS)
            pos += d * seg.speedMps
            remaining -= d
        }
        if (remaining > 0 && config.endless) pos += remaining * (trajectory.lastOrNull()?.speedMps ?: 0.0)
        return pos
    }

    private fun palletPositions(): Sequence<Double> =
        if (config.endless) generateSequence(config.firstPalletM) { it + config.palletPitchM }
        else (0 until config.palletCount).asSequence().map { config.firstPalletM + it * config.palletPitchM }

    /** Normalized image x of a line position [p] seen from camera position [c]. */
    fun imageX(p: Double, c: Double): Float = (0.5 - config.imageDirection * (p - c) / config.viewWidthM).toFloat()

    /**
     * Pallets whose centre ends up on the other side of a count line at [lineX] between the
     * start and the end of the trajectory: the correct answer for this scenario.
     */
    fun expectedCount(lineX: Float = 0.5f): Int {
        val offset = (0.5 - lineX) * config.viewWidthM * config.imageDirection
        val start = cameraPosition(0.0)
        val end = cameraPosition(durationS)
        val lo = minOf(start, end)
        val hi = maxOf(start, end)
        return palletPositions().takeWhile { it - offset <= hi + config.palletPitchM }
            .count { p -> (p - offset) > lo && (p - offset) <= hi }
    }

    /** Ground-truth boxes (visible part of each pallet) for camera position [c]. */
    fun groundTruth(c: Double): List<Box> {
        val w = (config.palletLengthM / config.viewWidthM).toFloat()
        val reach = config.viewWidthM + config.palletLengthM
        val result = ArrayList<Box>()
        for (p in palletPositions()) {
            val rel = p - c
            if (rel > reach) break // positions are ascending: nothing further can be visible
            if (rel < -reach) continue
            val visible = Box.fromCenter(imageX(p, c), config.boxCenterY, w, config.boxHeight).clippedToUnit()
            if (!visible.isValid || visible.width / w < config.minVisibleFraction) continue
            result += visible
        }
        return result
    }

    /** Detections for one moment in time using [rng] for the noise model. */
    fun detectionsAt(t: Double, rng: Random): List<Detection> {
        if (config.blackouts.any { t >= it.startS && t < it.endS }) return emptyList()
        val c = cameraPosition(t)
        val bob = (config.bobAmplitude * sin(2 * PI * config.bobHz * t)).toFloat()
        val dets = ArrayList<Detection>()
        for (gt in groundTruth(c)) {
            if (rng.nextDouble() < config.missProbability) continue
            val conf = if (rng.nextDouble() < config.lowConfidenceProbability) {
                0.2f + 0.25f * rng.nextFloat()
            } else {
                0.6f + 0.35f * rng.nextFloat()
            }
            val j = config.jitter
            val box = Box(
                gt.left + j * gaussian(rng),
                gt.top + bob + j * gaussian(rng),
                gt.right + j * gaussian(rng),
                gt.bottom + bob + j * gaussian(rng),
            ).clippedToUnit()
            if (box.isValid) dets += Detection(box, conf)
        }
        if (config.farRowSlivers) {
            // Far-row pallets peeking through the gaps between neighbouring near-row stacks.
            val positions = palletPositions().iterator()
            var previous = if (positions.hasNext()) positions.next() else null
            while (previous != null && positions.hasNext()) {
                val next = positions.next()
                val gap = (previous + next) / 2
                previous = next
                val rel = gap - c
                if (rel > config.viewWidthM) break
                if (rel < -config.viewWidthM) continue
                val x = imageX(gap, c)
                if (x < 0.1f || x > 0.9f) continue
                val box = Box.fromCenter(x, config.boxCenterY - 0.06f + bob, 0.08f, config.boxHeight * 0.65f)
                dets += Detection(box, 0.55f + 0.25f * rng.nextFloat())
            }
        }
        if (rng.nextDouble() < config.falsePositivesPerFrame) {
            val w = 0.05f + 0.2f * rng.nextFloat()
            val h = 0.05f + 0.07f * rng.nextFloat()
            val x = 0.05f + 0.9f * rng.nextFloat()
            val y = 0.35f + 0.45f * rng.nextFloat()
            dets += Detection(Box.fromCenter(x, y, w, h).clippedToUnit(), 0.3f + 0.4f * rng.nextFloat())
        }
        return dets
    }

    /** All detector frames of the scenario at [SimConfig.fps]. */
    fun frames(): List<DetectionFrame> {
        val rng = Random(config.seed)
        val n = floor(durationS * config.fps).toInt() + 1
        return List(n) { i ->
            val t = i / config.fps
            DetectionFrame(
                timestampNanos = (t * 1e9).toLong(),
                detections = detectionsAt(t, rng),
                frameWidth = 1280,
                frameHeight = 960,
                frameIndex = i.toLong(),
            )
        }
    }

    companion object {
        /** Walk past every pallet at a steady 0.8 m/s with some margin on both ends. */
        fun defaultTrajectory(config: SimConfig): List<Segment> {
            val length = config.firstPalletM + (config.palletCount - 1) * config.palletPitchM + 2.5
            val speed = 0.8
            return listOf(Segment(length / speed, speed))
        }

        private fun gaussian(rng: Random): Float {
            // Irwin–Hall approximation; plenty for jitter.
            var s = 0.0
            repeat(6) { s += rng.nextDouble() }
            return ((s - 3.0) / 0.7071).toFloat()
        }
    }
}
