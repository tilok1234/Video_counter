package com.palletcounter.core.replay

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Regression test on REAL model output: training/predict_video.py ran an exported LiteRT
 * model (trained on synthetic images) over a synthetic walk video with 12 pallets, a
 * 2 s pause and a 1.5 s walk back. Also checks the Python log writer is compatible with the
 * Kotlin parser.
 */
class PythonLogReplayTest {
    @Test
    fun syntheticWalkWithPauseAndReversalCountsTwelve() {
        val file = File(javaClass.getResource("/replay/synthetic_walk_python_predict.jsonl")!!.toURI())
        val log = file.bufferedReader().use { DetectionLog.parse(it) }
        assertTrue(log.errors.isEmpty(), log.errors.toString())
        assertEquals("python-predict", log.header?.source)
        assertEquals(12, log.header?.expectedCount)
        val result = Replay.run(log)
        assertEquals(12, result.count)
        assertEquals(true, result.matchesExpected)
        // The walk back must have produced an un-count and a re-count, not a duplicate.
        assertTrue(result.stats.counter.leftToRight >= 1, "expected at least one reverse crossing")
    }
}
