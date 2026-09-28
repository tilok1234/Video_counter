package com.palletcounter.core.detection

import com.palletcounter.core.geometry.Box
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.float
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Decodes raw output tensors recorded from REAL exported LiteRT models (trained on
 * synthetic images, see training/make_golden_fixture.py) and checks that the app's decoder
 * reproduces both the Python reference decoder and Ultralytics' own predictions.
 */
class YoloGoldenTest {
    private val dir = File(javaClass.getResource("/golden")!!.toURI())

    @Test
    fun allGoldenFixturesDecodeLikeTheReferences() {
        val fixtures = dir.listFiles { f -> f.name.endsWith(".json") }!!.sortedBy { it.name }
        assertTrue(fixtures.isNotEmpty(), "no golden fixtures found")
        for (f in fixtures) check(f)
    }

    private fun check(metaFile: File) {
        val name = metaFile.name.removeSuffix(".json")
        val meta = Json.parseToJsonElement(metaFile.readText()).jsonObject
        val shape = meta.getValue("shape").jsonArray.map { it.jsonPrimitive.int }.toIntArray()
        val bytes = File(dir, "$name.output.bin").readBytes()
        val floats = FloatArray(bytes.size / 4)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(floats)
        val lb = Letterbox(
            meta.getValue("src_width").jsonPrimitive.int,
            meta.getValue("src_height").jsonPrimitive.int,
            meta.getValue("input_width").jsonPrimitive.int,
            meta.getValue("input_height").jsonPrimitive.int,
        )
        val sidecar = ModelSidecar(
            outputFormat = meta.getValue("output_format").jsonPrimitive.content,
            coordinates = meta.getValue("coordinates").jsonPrimitive.content,
        )
        val conf = meta.getValue("conf").jsonPrimitive.float
        val dets = YoloDecoder(sidecar.toDecoderConfig(confidenceThreshold = conf)).decode(floats, shape, lb)
            .sortedByDescending { it.confidence }

        val python = meta.getValue("python_decoded").jsonArray.map { it.jsonArray.map { v -> v.jsonPrimitive.float } }
            .sortedByDescending { it[4] }
        assertEquals(python.size, dets.size, "$name: detection count vs python reference")
        for ((d, p) in dets.zip(python)) {
            assertEquals(p[0], d.box.left, 2e-3f, "$name left")
            assertEquals(p[1], d.box.top, 2e-3f, "$name top")
            assertEquals(p[2], d.box.right, 2e-3f, "$name right")
            assertEquals(p[3], d.box.bottom, 2e-3f, "$name bottom")
            assertEquals(p[4], d.confidence, 1e-4f, "$name score")
        }

        val reference = meta.getValue("ultralytics").jsonArray.map { it.jsonArray.map { v -> v.jsonPrimitive.float } }
        for (r in reference) {
            val refBox = Box(r[0], r[1], r[2], r[3])
            val best = dets.maxOfOrNull { it.box.iou(refBox) } ?: 0f
            // Loose on purpose: end-to-end heads differ slightly from the PyTorch NMS head;
            // a decoding error (layout, coordinates, letterbox) gives IoU near 0.
            assertTrue(best >= 0.8f, "$name: ultralytics box $refBox not reproduced (best IoU $best)")
        }
    }
}
