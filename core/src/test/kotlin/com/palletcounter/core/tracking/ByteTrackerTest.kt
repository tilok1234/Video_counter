package com.palletcounter.core.tracking

import com.palletcounter.core.detection.Detection
import com.palletcounter.core.detection.DetectionFrame
import com.palletcounter.core.geometry.Box
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class ByteTrackerTest {
    private val frameNanos = 100_000_000L // 10 detector frames per second

    private fun frame(i: Int, vararg dets: Detection) =
        DetectionFrame(timestampNanos = i * frameNanos, detections = dets.toList(), frameIndex = i.toLong())

    private fun det(cx: Float, conf: Float = 0.9f, w: Float = 0.3f, cy: Float = 0.7f, h: Float = 0.1f) =
        Detection(Box.fromCenter(cx, cy, w, h), conf)

    @Test
    fun steadyObjectKeepsOneIdAndConfirms() {
        val tracker = ByteTracker()
        val ids = HashSet<Int>()
        var last: TrackerOutput? = null
        for (i in 0 until 20) {
            last = tracker.update(frame(i, det(0.9f - i * 0.03f)))
            last.tracks.forEach { ids += it.id }
        }
        assertEquals(setOf(1), ids)
        assertEquals(TrackState.CONFIRMED, last!!.tracks.single().state)
        // velocity estimate converges to -0.3 frame widths per second
        assertEquals(-0.3f, last.tracks.single().velocityX, 0.05f)
    }

    @Test
    fun trackStaysTentativeUntilMinHits() {
        val tracker = ByteTracker(TrackerConfig(minHitsToConfirm = 3))
        assertEquals(TrackState.TENTATIVE, tracker.update(frame(0, det(0.5f))).tracks.single().state)
        assertEquals(TrackState.TENTATIVE, tracker.update(frame(1, det(0.5f))).tracks.single().state)
        assertEquals(TrackState.CONFIRMED, tracker.update(frame(2, det(0.5f))).tracks.single().state)
    }

    @Test
    fun twoAdjacentObjectsKeepSeparateIds() {
        val tracker = ByteTracker()
        val seen = HashMap<Int, MutableList<Float>>()
        for (i in 0 until 25) {
            val x = 0.8f - i * 0.02f
            val out = tracker.update(frame(i, det(x), det(x - 0.3f)))
            out.tracks.filter { it.updatedThisFrame }.forEach { seen.getOrPut(it.id) { mutableListOf() } += it.box.centerX }
        }
        assertEquals(setOf(1, 2), seen.keys)
        // No swap: the right object stays right of the left one for its whole life.
        val a = seen.getValue(1)
        val b = seen.getValue(2)
        for (k in a.indices) assertTrue(a[k] > b[k])
    }

    @Test
    fun shortOcclusionRecoversSameId() {
        val tracker = ByteTracker()
        val v = -0.03f
        for (i in 0 until 10) tracker.update(frame(i, det(0.9f + v * i)))
        // 6 frames (0.6 s) without detections
        for (i in 10 until 16) {
            val out = tracker.update(frame(i))
            assertEquals(TrackState.LOST, out.tracks.single().state)
        }
        val out = tracker.update(frame(16, det(0.9f + v * 16)))
        val t = out.tracks.single()
        assertEquals(1, t.id)
        assertEquals(TrackState.CONFIRMED, t.state)
        assertEquals(1, t.recoveries)
        assertTrue(out.born.isEmpty())
    }

    @Test
    fun longOcclusionStartsNewTrack() {
        val tracker = ByteTracker(TrackerConfig(lostTimeoutMillis = 500))
        for (i in 0 until 10) tracker.update(frame(i, det(0.5f)))
        var removed = false
        for (i in 10 until 20) removed = removed or tracker.update(frame(i)).removed.any { it.id == 1 }
        assertTrue(removed)
        val out = tracker.update(frame(20, det(0.5f)))
        assertNotEquals(1, out.tracks.single().id)
    }

    @Test
    fun lowScoreDetectionsKeepConfirmedTrackAlive() {
        val tracker = ByteTracker()
        for (i in 0 until 5) tracker.update(frame(i, det(0.5f, conf = 0.9f)))
        for (i in 5 until 15) {
            val out = tracker.update(frame(i, det(0.5f, conf = 0.3f)))
            val t = out.tracks.single()
            assertEquals(1, t.id)
            assertEquals(TrackState.CONFIRMED, t.state)
            assertEquals(MatchStage.LOW, t.lastMatchStage)
        }
    }

    @Test
    fun lowScoreDetectionsNeverStartTracks() {
        val tracker = ByteTracker()
        for (i in 0 until 10) assertTrue(tracker.update(frame(i, det(0.5f, conf = 0.3f))).tracks.isEmpty())
    }

    @Test
    fun lostTrackFollowsCameraMotionWhenCameraStops() {
        // Two pallets move left together; then the camera stops. Pallet A is hidden while the
        // camera stops and reappears where it stopped. Without motion compensation its
        // prediction would keep drifting left and it would come back with a new id.
        fun run(compensate: Boolean): Int {
            val tracker = ByteTracker(TrackerConfig(motionCompensateLostTracks = compensate))
            val v = -0.03f
            var ax = 0.95f
            var bx = 0.55f
            for (i in 0 until 10) {
                tracker.update(frame(i, det(ax), det(bx)))
                ax += v; bx += v
            }
            // Camera stops. A is hidden for 1 s, B stays visible and stationary.
            for (i in 10 until 20) tracker.update(frame(i, det(bx)))
            val out = tracker.update(frame(20, det(ax), det(bx)))
            return out.tracks.first { it.updatedThisFrame && kotlin.math.abs(it.box.centerX - ax) < 0.1f }.id
        }
        assertEquals(1, run(compensate = true))
        assertNotEquals(1, run(compensate = false))
    }

    @Test
    fun motionEstimateReflectsCameraMotion() {
        val tracker = ByteTracker()
        var out: TrackerOutput? = null
        for (i in 0 until 15) out = tracker.update(frame(i, det(0.9f - 0.04f * i), det(0.5f - 0.04f * i)))
        val m = out!!.motion
        assertTrue(m.valid)
        assertEquals(-0.4f, m.vx, 0.05f)
    }

    @Test
    fun duplicateOverlappingDetectionDoesNotSpawnSecondTrack() {
        val tracker = ByteTracker()
        for (i in 0 until 5) tracker.update(frame(i, det(0.5f)))
        val out = tracker.update(frame(5, det(0.5f), det(0.505f, conf = 0.8f)))
        assertEquals(1, out.tracks.size)
    }
}
