package com.palletcounter.core.replay

import com.palletcounter.core.detection.Detection
import com.palletcounter.core.detection.DetectionFrame
import com.palletcounter.core.detection.DetectorInfo
import com.palletcounter.core.geometry.Box
import com.palletcounter.core.pipeline.PipelineConfig
import com.palletcounter.core.pipeline.ScanPipeline
import com.palletcounter.core.session.ScanSettings
import com.palletcounter.core.sim.SimConfig
import com.palletcounter.core.sim.SyntheticLineScene
import java.io.StringWriter
import java.util.Locale
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DetectionLogTest {
    @Test
    fun roundTripPreservesFramesAndHeader() {
        val frames = listOf(
            DetectionFrame(1_000L, listOf(Detection(Box(0.1f, 0.2f, 0.4f, 0.35f), 0.912f, 0)), 1280, 960, 21.5f, 0),
            DetectionFrame(101_000_000L, emptyList(), 1280, 960, 19f, 1),
        )
        val sw = StringWriter()
        DetectionLogWriter(sw).use { w ->
            w.header(LogHeader(source = "test", detector = DetectorInfo("m", "tflite-yolo"), settings = ScanSettings(), pipeline = PipelineConfig(), expectedCount = 3))
            frames.forEach(w::frame)
            w.footer(LogFooter(count = 2, adjustment = 1, accepted = true))
        }
        val parsed = DetectionLog.parse(sw.toString().lineSequence())
        assertTrue(parsed.errors.isEmpty(), parsed.errors.toString())
        assertEquals(3, parsed.header?.expectedCount)
        assertEquals("m", parsed.header?.detector?.name)
        assertEquals(PipelineConfig(), parsed.header?.pipeline)
        assertEquals(2, parsed.footer?.count)
        assertEquals(2, parsed.frames.size)
        val d = parsed.frames[0].detections.single()
        assertEquals(0.1f, d.box.left, 1e-4f)
        assertEquals(0.35f, d.box.bottom, 1e-4f)
        assertEquals(0.912f, d.confidence, 1e-3f)
        assertEquals(1280, parsed.frames[0].frameWidth)
        assertEquals(21.5f, parsed.frames[0].inferenceMillis, 1e-3f)
        assertEquals(101_000_000L, parsed.frames[1].timestampNanos)
    }

    @Test
    fun numberFormattingIgnoresDefaultLocale() {
        val previous = Locale.getDefault()
        try {
            Locale.setDefault(Locale.forLanguageTag("nb-NO")) // uses decimal commas
            val line = DetectionLog.formatFrame(DetectionFrame(5, listOf(Detection(Box(0.25f, 0.5f, 0.75f, 0.6f), 0.5f))))
            assertTrue(!line.contains(",5,") && line.contains("0.25"), line)
            assertEquals(1, DetectionLog.parse(sequenceOf(line)).frames.size)
        } finally {
            Locale.setDefault(previous)
        }
    }

    @Test
    fun malformedLinesAreReportedNotThrown() {
        val parsed = DetectionLog.parse(sequenceOf("{\"type\":\"frame\",\"t\":1,\"d\":[]}", "not json", "", "{\"type\":\"what\"}"))
        assertEquals(1, parsed.frames.size)
        assertEquals(2, parsed.errors.size)
    }

    @Test
    fun replayOfRecordedLogReproducesLiveCount() {
        // Simulate what the app does: process frames live and log them, then replay the log.
        val scene = SyntheticLineScene(SimConfig(seed = 7, missProbability = 0.15, falsePositivesPerFrame = 0.1))
        val live = ScanPipeline()
        val sw = StringWriter()
        DetectionLogWriter(sw).use { w ->
            w.header(LogHeader(source = "test", pipeline = live.config, expectedCount = scene.expectedCount()))
            for (f in scene.frames()) {
                live.process(f)
                w.frame(f)
            }
            w.footer(LogFooter(count = live.count))
        }
        val result = Replay.run(DetectionLog.parse(sw.toString().lineSequence()))
        assertEquals(live.count, result.count)
        assertEquals(result.recordedCount, result.count)
        assertEquals(true, result.matchesExpected)
    }
}
