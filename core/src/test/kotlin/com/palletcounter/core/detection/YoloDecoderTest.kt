package com.palletcounter.core.detection

import com.palletcounter.core.geometry.Box
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class YoloDecoderTest {
    private val lb = Letterbox(1280, 960, 640, 640) // content 640x480 at padTop 80

    /** Builds a channels-first [1, 4+nc, n] tensor with the given (anchor -> values) entries. */
    private fun channelsFirst(n: Int, nc: Int, entries: Map<Int, FloatArray>): FloatArray {
        val c = 4 + nc
        val out = FloatArray(c * n)
        for ((anchor, v) in entries) for (ch in 0 until c) out[ch * n + anchor] = v[ch]
        return out
    }

    @Test
    fun layoutAutoDetection() {
        assertEquals(YoloOutputFormat.RAW_CHANNELS_FIRST, YoloLayout.resolve(intArrayOf(1, 5, 8400), 640, 640).format)
        assertEquals(YoloOutputFormat.RAW_CHANNELS_LAST, YoloLayout.resolve(intArrayOf(1, 8400, 5), 640, 640).format)
        assertEquals(YoloOutputFormat.END_TO_END, YoloLayout.resolve(intArrayOf(1, 300, 6), 640, 640).format)
        // Two-class raw model has 6 channels but 8400 anchors: must not be mistaken for end-to-end.
        assertEquals(YoloOutputFormat.RAW_CHANNELS_FIRST, YoloLayout.resolve(intArrayOf(1, 6, 8400), 640, 640).format)
        assertEquals(YoloOutputFormat.RAW_CHANNELS_LAST, YoloLayout.resolve(intArrayOf(1, 8400, 6), 640, 640).format)
        assertEquals(2100, YoloLayout.expectedAnchorCount(320, 320))
        assertEquals(6300, YoloLayout.expectedAnchorCount(640, 480))
        assertFailsWith<IllegalArgumentException> { YoloLayout.resolve(intArrayOf(2, 5, 8400), 640, 640) }
    }

    @Test
    fun decodesNormalizedChannelsFirstIntoSourceCoordinates() {
        val n = 8400
        // A box centred in the letterboxed content: input px (320, 320) size 320x96.
        val v = floatArrayOf(0.5f, 0.5f, 0.5f, 0.15f, 0.9f)
        val out = channelsFirst(n, 1, mapOf(10 to v))
        val dets = YoloDecoder(YoloDecoderConfig(confidenceThreshold = 0.25f))
            .decode(out, intArrayOf(1, 5, n), lb)
        assertEquals(1, dets.size)
        val box = dets[0].box
        // x: 160..480 of 640 wide content -> 0.25..0.75
        assertEquals(0.25f, box.left, 1e-4f)
        assertEquals(0.75f, box.right, 1e-4f)
        // y: 272..368 minus pad 80 -> 192..288 of 480 -> 0.4..0.6
        assertEquals(0.4f, box.top, 1e-4f)
        assertEquals(0.6f, box.bottom, 1e-4f)
        assertEquals(0.9f, dets[0].confidence, 1e-6f)
    }

    @Test
    fun pixelCoordinatesAreDetectedAutomatically() {
        val n = 8400
        val v = floatArrayOf(320f, 320f, 320f, 96f, 0.8f)
        val out = channelsFirst(n, 1, mapOf(3 to v))
        val dets = YoloDecoder().decode(out, intArrayOf(1, 5, n), lb)
        assertEquals(1, dets.size)
        assertEquals(0.25f, dets[0].box.left, 1e-4f)
        assertEquals(0.6f, dets[0].box.bottom, 1e-4f)
    }

    @Test
    fun thresholdAndNmsAreApplied() {
        val n = 8400
        val out = channelsFirst(
            n, 1,
            mapOf(
                0 to floatArrayOf(0.5f, 0.5f, 0.3f, 0.1f, 0.9f),
                1 to floatArrayOf(0.505f, 0.5f, 0.3f, 0.1f, 0.7f), // duplicate of anchor 0
                2 to floatArrayOf(0.2f, 0.5f, 0.2f, 0.1f, 0.05f), // below threshold
            ),
        )
        val dets = YoloDecoder(YoloDecoderConfig(confidenceThreshold = 0.1f)).decode(out, intArrayOf(1, 5, n), lb)
        assertEquals(1, dets.size)
        assertEquals(0.9f, dets[0].confidence, 1e-6f)
    }

    @Test
    fun targetClassFilterSelectsOnlyPalletClass() {
        val n = 8400
        // class 0 = other object with high score, class 1 = pallet with moderate score
        val out = channelsFirst(n, 2, mapOf(5 to floatArrayOf(0.5f, 0.5f, 0.3f, 0.1f, 0.95f, 0.6f)))
        val dets = YoloDecoder(YoloDecoderConfig(targetClassIds = setOf(1), confidenceThreshold = 0.25f))
            .decode(out, intArrayOf(1, 6, n), lb)
        assertEquals(1, dets.size)
        assertEquals(1, dets[0].classId)
        assertEquals(0.6f, dets[0].confidence, 1e-6f)
    }

    @Test
    fun decodesEndToEndRows() {
        val rows = 300
        val out = FloatArray(rows * 6)
        fun row(i: Int, vararg v: Float) = v.forEachIndexed { k, x -> out[i * 6 + k] = x }
        row(0, 0.25f, 80f / 640f + 0.4f * 0.75f, 0.75f, 80f / 640f + 0.6f * 0.75f, 0.88f, 0f)
        row(1, 0.1f, 0.3f, 0.2f, 0.4f, 0.01f, 0f) // padding row, low score
        val dets = YoloDecoder(YoloDecoderConfig(confidenceThreshold = 0.25f)).decode(out, intArrayOf(1, rows, 6), lb)
        assertEquals(1, dets.size)
        val b = dets[0].box
        assertEquals(0.25f, b.left, 1e-4f)
        assertEquals(0.4f, b.top, 1e-4f)
        assertEquals(0.6f, b.bottom, 1e-4f)
    }

    @Test
    fun boxesInPaddingAreClippedOrDropped() {
        val n = 8400
        // Entirely inside the top padding band (y < 80px) -> clipped to zero height -> dropped
        val out = channelsFirst(n, 1, mapOf(0 to floatArrayOf(0.5f, 20f / 640f, 0.2f, 20f / 640f, 0.9f)))
        val dets = YoloDecoder().decode(out, intArrayOf(1, 5, n), lb)
        assertTrue(dets.isEmpty())
    }

    @Test
    fun sidecarMapsToDecoderConfig() {
        val s = ModelSidecar.parse(
            """{"name":"m","output_format":"yolo_end2end","coordinates":"normalized",
               "classes":["other","eur_pallet_base"],"unknown_field":1}""",
        )
        val cfg = s.toDecoderConfig(confidenceThreshold = 0.2f)
        assertEquals(YoloOutputFormat.END_TO_END, cfg.format)
        assertEquals(CoordinateSpace.NORMALIZED, cfg.coordinates)
        assertEquals(setOf(1), cfg.targetClassIds)
        assertEquals(setOf(0), ModelSidecar().targetClassIds())
    }

    @Test
    fun boxRoundTripThroughDecoderMatchesLetterbox() {
        val src = Box(0.1f, 0.55f, 0.45f, 0.7f)
        val px = lb.sourceToInputPixels(src)
        val n = 8400
        val v = floatArrayOf(
            px.centerX / 640f, px.centerY / 640f, px.width / 640f, px.height / 640f, 0.77f,
        )
        val det = YoloDecoder().decode(channelsFirst(n, 1, mapOf(42 to v)), intArrayOf(1, 5, n), lb).single()
        assertEquals(src.left, det.box.left, 1e-4f)
        assertEquals(src.top, det.box.top, 1e-4f)
        assertEquals(src.right, det.box.right, 1e-4f)
        assertEquals(src.bottom, det.box.bottom, 1e-4f)
    }
}
