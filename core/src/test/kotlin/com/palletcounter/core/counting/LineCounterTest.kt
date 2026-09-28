package com.palletcounter.core.counting

import com.palletcounter.core.detection.Detection
import com.palletcounter.core.detection.DetectionFrame
import com.palletcounter.core.geometry.Box
import com.palletcounter.core.pipeline.PipelineConfig
import com.palletcounter.core.pipeline.ScanPipeline
import com.palletcounter.core.tracking.TrackerConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Counting rules exercised through the full ROI → tracker → counter pipeline with
 * hand-written detection sequences (10 detector frames per second).
 */
class LineCounterTest {
    private val stepNanos = 100_000_000L

    private class Feed(val pipeline: ScanPipeline) {
        var frame = 0
        fun push(vararg boxes: Box?, conf: Float = 0.9f) {
            val dets = boxes.filterNotNull().map { Detection(it, conf) }
            pipeline.process(DetectionFrame(frame * 100_000_000L, dets, frameIndex = frame.toLong()))
            frame++
        }
        fun empty(n: Int) = repeat(n) { push() }
    }

    private fun pallet(cx: Float, w: Float = 0.3f) = Box.fromCenter(cx, 0.6f, w, 0.1f)

    private fun feed(config: PipelineConfig = PipelineConfig()) = Feed(ScanPipeline(config))

    /** Moves one pallet from x0 to x1 in [steps] frames. */
    private fun Feed.sweep(x0: Float, x1: Float, steps: Int) {
        for (i in 0..steps) push(pallet(x0 + (x1 - x0) * i / steps))
    }

    @Test
    fun oneTrackedObjectCrossingIsCountedOnce() {
        val f = feed()
        f.sweep(0.9f, 0.1f, 30)
        assertEquals(1, f.pipeline.count)
        assertEquals(1, f.pipeline.allEvents.size)
    }

    @Test
    fun repeatedDetectionsWithoutCrossingDoNotCount() {
        val f = feed()
        repeat(50) { f.push(pallet(0.75f)) }
        assertEquals(0, f.pipeline.count)
    }

    @Test
    fun palletParkedOnTheLineWithJitterDoesNotCount() {
        val f = feed()
        f.sweep(0.9f, 0.52f, 10)
        // Camera stopped: box centre jitters within the ±0.04 hysteresis band.
        for (i in 0 until 60) f.push(pallet(0.5f + if (i % 2 == 0) 0.03f else -0.03f))
        assertEquals(0, f.pipeline.count)
        f.sweep(0.5f, 0.1f, 10)
        assertEquals(1, f.pipeline.count)
    }

    @Test
    fun temporaryDisappearanceDoesNotCreateDuplicate() {
        val f = feed()
        f.sweep(0.9f, 0.4f, 10) // crossed the line, counted
        assertEquals(1, f.pipeline.count)
        f.empty(6) // 0.6 s detector failure
        f.sweep(0.4f - 0.03f * 6, 0.05f, 6)
        assertEquals(1, f.pipeline.count)
    }

    @Test
    fun newObjectCrossingLineAddsOne() {
        val f = feed()
        f.sweep(0.9f, 0.1f, 20)
        f.sweep(0.9f, 0.1f, 20)
        assertEquals(2, f.pipeline.count)
    }

    @Test
    fun alreadyCountedObjectCrossingAgainIsNotDuplicated() {
        val f = feed()
        f.sweep(0.9f, 0.3f, 12) // counted
        f.sweep(0.3f, 0.7f, 8) // camera reverses: pallet crosses back -> un-counted
        assertEquals(0, f.pipeline.count)
        f.sweep(0.7f, 0.1f, 12) // crosses again
        assertEquals(1, f.pipeline.count)
        assertEquals(3, f.pipeline.allEvents.size) // +1, -1, +1: never 2
    }

    @Test
    fun countingWorksInBothWalkingDirections() {
        val f = feed()
        f.sweep(0.1f, 0.9f, 20)
        f.sweep(0.1f, 0.9f, 20)
        f.sweep(0.1f, 0.9f, 20)
        assertEquals(3, f.pipeline.count)
        assertEquals(3, f.pipeline.netCount)
    }

    @Test
    fun tentativeTrackThatDiesBeforeConfirmationIsNotCounted() {
        val f = feed()
        // Only 2 detections (min hits = 3), straddling the line.
        f.push(pallet(0.6f))
        f.push(pallet(0.4f))
        f.empty(20)
        assertEquals(0, f.pipeline.count)
    }

    @Test
    fun fastPalletCrossingBeforeConfirmationIsCountedOnConfirmation() {
        val f = feed()
        // Crosses the band between its 1st and 2nd detection, i.e. while still tentative
        // (the filtered centre after the 2nd detection is ~0.44, left of the band).
        f.push(pallet(0.62f, w = 0.4f))
        f.push(pallet(0.42f, w = 0.4f))
        assertEquals(0, f.pipeline.count)
        f.push(pallet(0.30f, w = 0.4f)) // 3rd hit confirms the track -> held crossing applied
        assertEquals(1, f.pipeline.count)
        assertTrue(f.pipeline.allEvents.single().delayedMillis > 0)
        f.push(pallet(0.20f, w = 0.4f))
        assertEquals(1, f.pipeline.count)
    }

    @Test
    fun detectionsOutsideRoiAreIgnored() {
        val f = feed(PipelineConfig(roi = Box(0f, 0.5f, 1f, 0.9f)))
        // Background pallet high in the frame (y = 0.2) crossing the line.
        for (i in 0..20) f.push(Box.fromCenter(0.9f - 0.04f * i, 0.2f, 0.2f, 0.06f))
        assertEquals(0, f.pipeline.count)
    }

    @Test
    fun trackLostAcrossTheLineIsStitchedAndCountedOnce() {
        // Detector loses the pallet just before the line for longer than the tracker keeps
        // lost tracks; it is re-detected past the line. Stitching must link the two so the
        // crossing is neither missed nor double counted.
        val cfg = PipelineConfig(tracker = TrackerConfig(lostTimeoutMillis = 400))
        val f = feed(cfg)
        val v = -0.03f
        var x = 0.9f
        repeat(12) { f.push(pallet(x)); x += v } // x ends ~0.54, still right of the band
        repeat(8) { f.push(); x += v } // 0.8 s gap: track removed after 0.4 s
        repeat(10) { f.push(pallet(x)); x += v }
        assertEquals(1, f.pipeline.count)
        assertTrue(f.pipeline.latest!!.stats.counter.stitches >= 1)
    }

    @Test
    fun idSwitchAfterCountingDoesNotDoubleCount() {
        val cfg = PipelineConfig(tracker = TrackerConfig(lostTimeoutMillis = 300))
        val f = feed(cfg)
        f.sweep(0.9f, 0.35f, 12)
        assertEquals(1, f.pipeline.count)
        f.empty(8)
        f.sweep(0.25f, 0.05f, 8)
        assertEquals(1, f.pipeline.count)
    }

    @Test
    fun movingTheLineMidScanNeverCreatesCrossings() {
        val p = ScanPipeline()
        val f = Feed(p)
        repeat(10) { f.push(pallet(0.6f)) }
        p.updateConfig(p.config.copy(counter = p.config.counter.copy(lineX = 0.7f)))
        repeat(10) { f.push(pallet(0.6f)) }
        assertEquals(0, p.count)
    }

    @Test
    fun sizeFilterIgnoresSmallFarRowBoxesOnceReferenceExists() {
        val cfg = PipelineConfig(counter = CounterConfig(relativeSizeFilter = 0.8f, sizeFilterMinReference = 2))
        val f = feed(cfg)
        repeat(2) { f.sweep(0.9f, 0.1f, 16) } // two normal pallets, height 0.1 (reference)
        // far-row box: 60 % height
        for (i in 0..16) f.push(Box.fromCenter(0.9f - 0.05f * i, 0.55f, 0.08f, 0.06f))
        assertEquals(2, f.pipeline.count)
    }
}
