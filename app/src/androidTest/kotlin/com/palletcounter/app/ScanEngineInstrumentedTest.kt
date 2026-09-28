package com.palletcounter.app

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.palletcounter.app.data.AppSettings
import com.palletcounter.app.data.DetectorMode
import com.palletcounter.app.debug.CaptureStore
import com.palletcounter.app.detection.FrameInput
import com.palletcounter.app.detection.ModelManager
import com.palletcounter.app.scan.ScanEngine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** The scan engine (worker thread, pipeline, thumbnails) on a device, fed by the SIMULATION detector. */
@RunWith(AndroidJUnit4::class)
class ScanEngineInstrumentedTest {
    @Test
    fun simulatedSweepIsCountedByTheEngine() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val settings = AppSettings(detectorMode = DetectorMode.SIMULATION, recordDetectionLogs = false)
        val engine = ScanEngine(ctx, ModelManager(ctx), settings, "test", logStore = null, captures = CaptureStore(ctx))
        engine.start()
        assertTrue(engine.awaitReady())
        val frame = Bitmap.createBitmap(64, 48, Bitmap.Config.ARGB_8888)
        for (i in 0 until 300) { // 30 s at 10 detector frames per second
            engine.processBlocking(FrameInput(frame, 64, 48, 0, i * 100_000_000L))
        }
        val result = engine.finish()
        engine.close()
        // Endless simulated line walked at 0.7 m/s: pallets at 2.5 m + k * 1.25 m cross the
        // centre line within 30 s for k = 0..14.
        assertTrue("count ${result.detectedCount}", result.detectedCount in 14..16)
        assertEquals(result.detectedCount, result.pallets.size)
        assertTrue(result.pallets.all { it.thumbnail != null })
        assertTrue(result.detector?.isSimulation == true)
    }
}
