package com.palletcounter.app

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.palletcounter.app.data.AppSettings
import com.palletcounter.app.data.DetectorMode
import com.palletcounter.app.debug.CaptureStore
import com.palletcounter.app.debug.LogStore
import com.palletcounter.app.detection.FrameInput
import com.palletcounter.app.detection.ModelManager
import com.palletcounter.app.scan.ScanEngine
import com.palletcounter.core.replay.DetectionLog
import com.palletcounter.core.replay.Replay
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

    @Test
    fun closingTwiceAndUsingAClosedEngineIsHarmless() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val settings = AppSettings(detectorMode = DetectorMode.SIMULATION, recordDetectionLogs = false)
        val engine = ScanEngine(ctx, ModelManager(ctx), settings, "test", logStore = null, captures = CaptureStore(ctx))
        engine.start()
        assertTrue(engine.awaitReady())
        engine.finish()
        // FINISH followed by back / a second FINISH closes the engine twice.
        engine.close()
        engine.close()
        engine.resetCount()
        engine.submit(FrameInput(Bitmap.createBitmap(64, 48, Bitmap.Config.ARGB_8888), 64, 48, 0, 0L))
    }

    @Test
    fun resetStartsANewLogAndEachLogReplaysToItsRecordedCount() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val logStore = LogStore(ctx)
        val existing = logStore.list().toSet()
        val settings = AppSettings(detectorMode = DetectorMode.SIMULATION, recordDetectionLogs = true)
        val engine = ScanEngine(ctx, ModelManager(ctx), settings, "test", logStore = logStore, captures = CaptureStore(ctx))
        engine.start()
        assertTrue(engine.awaitReady())
        val frame = Bitmap.createBitmap(64, 48, Bitmap.Config.ARGB_8888)
        var t = 0L
        repeat(150) { engine.processBlocking(FrameInput(frame, 64, 48, 0, t)); t += 100_000_000L }
        val countBeforeReset = engine.state.value.snapshot?.count ?: 0
        assertTrue("count before reset $countBeforeReset", countBeforeReset > 0)
        engine.resetCount()
        repeat(150) { engine.processBlocking(FrameInput(frame, 64, 48, 0, t)); t += 100_000_000L }
        val result = engine.finish()
        engine.close()

        val logs = logStore.list().filter { it !in existing }.sortedBy { it.name }
        try {
            assertEquals(2, logs.size)
            val beforeReset = DetectionLog.parse(logs[0].bufferedReader())
            val afterReset = DetectionLog.parse(logs[1].bufferedReader())
            assertEquals(countBeforeReset, beforeReset.footer?.count)
            assertEquals(countBeforeReset, Replay.run(beforeReset).count)
            assertEquals(result.detectedCount, afterReset.footer?.count)
            assertEquals(result.detectedCount, Replay.run(afterReset).count)
            assertEquals(logs[1], result.logFile)
            assertEquals(150L, result.frames)
        } finally {
            logs.forEach { it.delete() }
        }
    }
}
