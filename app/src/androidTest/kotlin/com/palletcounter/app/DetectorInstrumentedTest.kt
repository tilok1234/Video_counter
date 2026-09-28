package com.palletcounter.app

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.palletcounter.app.data.Accelerator
import com.palletcounter.app.detection.FrameInput
import com.palletcounter.app.detection.TfliteYoloDetector
import com.palletcounter.core.detection.ModelSidecar
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.RandomAccessFile
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Runs LiteRT on the device with tiny purpose-built models (training/make_android_test_models.py):
 * one box at a known position whose score is the mean RED input value. Verifies input layout,
 * RGB order, normalisation, rotation, letterboxing and output decoding end to end.
 */
@RunWith(AndroidJUnit4::class)
class DetectorInstrumentedTest {
    private fun model(name: String): MappedByteBuffer {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val file = File(instrumentation.targetContext.cacheDir, name)
        instrumentation.context.assets.open(name).use { input -> file.outputStream().use { input.copyTo(it) } }
        return RandomAccessFile(file, "r").use { it.channel.map(FileChannel.MapMode.READ_ONLY, 0, it.length()) }
    }

    private fun solid(w: Int, h: Int, color: Int) = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).apply { eraseColor(color) }

    private fun checkModel(name: String) {
        TfliteYoloDetector(model(name), ModelSidecar(name = name), Accelerator.CPU, 2, decoderThreshold = 0.5f).use { det ->
            // 128x64 source -> 64x32 content + 16 px padding top/bottom in the 64x64 input.
            val red = det.detect(FrameInput(solid(128, 64, Color.RED), 128, 64, 0, 1L), 1L)
            assertEquals(1, red.detections.size)
            val b = red.detections[0].box
            assertEquals(0.25f, b.left, 0.02f)
            assertEquals(0.75f, b.right, 0.02f)
            assertEquals(0.25f, b.top, 0.03f)
            assertEquals(0.75f, b.bottom, 0.03f)
            // Red content (1.0) over half the input, grey padding (114/255) over the other half.
            assertEquals(0.7235f, red.detections[0].confidence, 0.03f)
            assertTrue(red.inferenceMillis >= 0f)

            val blue = det.detect(FrameInput(solid(128, 64, Color.BLUE), 128, 64, 0, 2L), 2L)
            assertTrue("blue frame must not score as red", blue.detections.isEmpty())

            // Portrait sensor buffer that must be rotated 90° to be upright: same result.
            val rotated = det.detect(FrameInput(solid(64, 128, Color.RED), 64, 128, 90, 3L), 3L)
            assertEquals(128, rotated.frameWidth)
            assertEquals(64, rotated.frameHeight)
            assertEquals(1, rotated.detections.size)
            assertEquals(0.25f, rotated.detections[0].box.top, 0.03f)
        }
    }

    @Test
    fun nchwModel() = checkModel("tiny_red_nchw.tflite")

    @Test
    fun nhwcModel() = checkModel("tiny_red_nhwc.tflite")

    @Test
    fun autoAcceleratorFallsBackGracefully() {
        TfliteYoloDetector(model("tiny_red_nchw.tflite"), ModelSidecar(), Accelerator.AUTO, 2, 0.5f).use { det ->
            assertTrue(det.info.accelerator, det.info.accelerator == "GPU" || det.info.accelerator.startsWith("CPU"))
            assertEquals(1, det.detect(FrameInput(solid(64, 64, Color.RED), 64, 64, 0, 1L), 1L).detections.size)
        }
    }
}
