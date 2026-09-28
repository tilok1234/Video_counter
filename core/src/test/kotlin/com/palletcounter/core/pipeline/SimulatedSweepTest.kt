package com.palletcounter.core.pipeline

import com.palletcounter.core.counting.CounterConfig
import com.palletcounter.core.sim.Blackout
import com.palletcounter.core.sim.Segment
import com.palletcounter.core.sim.SimConfig
import com.palletcounter.core.sim.SyntheticLineScene
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * End-to-end scenarios: a simulated walk along a line of 16 pallets with an imperfect
 * detector, run through the real pipeline. Each scenario targets failure cases from the
 * requirements (see docs/FAILURE_CASES.md). Several seeds are used so that a pass is not
 * a lucky draw of the noise.
 */
class SimulatedSweepTest {
    private val seeds = listOf(1L, 2L, 3L, 4L, 5L)

    private fun run(scene: SyntheticLineScene, config: PipelineConfig = PipelineConfig()): Int {
        val p = ScanPipeline(config)
        scene.frames().forEach { p.process(it) }
        return p.count
    }

    private fun assertCounts(
        name: String,
        config: PipelineConfig = PipelineConfig(),
        makeScene: (Long) -> SyntheticLineScene,
    ) {
        val results = seeds.map { seed ->
            val scene = makeScene(seed)
            scene.expectedCount() to run(scene, config)
        }
        val failures = results.filter { (expected, actual) -> expected != actual }
        assertTrue(failures.isEmpty(), "$name: (expected, actual) per seed = $results")
    }

    private fun walk(vararg segments: Pair<Double, Double>) = segments.map { Segment(it.first, it.second) }

    @Test
    fun steadyWalkCleanDetector() = assertCounts("steady") { seed ->
        SyntheticLineScene(
            SimConfig(seed = seed, missProbability = 0.0, lowConfidenceProbability = 0.0, falsePositivesPerFrame = 0.0),
        )
    }

    @Test
    fun steadyWalkNoisyDetector() = assertCounts("noisy") { seed ->
        SyntheticLineScene(
            SimConfig(seed = seed, missProbability = 0.2, lowConfidenceProbability = 0.2, jitter = 0.01f, falsePositivesPerFrame = 0.2),
        )
    }

    @Test
    fun detectorBlackouts() = assertCounts("blackouts") { seed ->
        SyntheticLineScene(
            SimConfig(seed = seed, blackouts = listOf(Blackout(6.0, 6.6), Blackout(14.0, 14.8), Blackout(20.0, 20.5))),
        )
    }

    @Test
    fun stopAndGo() = assertCounts("stop-and-go") { seed ->
        val cfg = SimConfig(seed = seed)
        SyntheticLineScene(cfg, walk(8.0 to 0.8, 4.0 to 0.0, 6.0 to 0.8, 3.0 to 0.0, 20.0 to 0.8))
    }

    @Test
    fun shortReversal() = assertCounts("short reversal") { seed ->
        SyntheticLineScene(SimConfig(seed = seed), walk(12.0 to 0.8, 3.0 to -0.6, 20.0 to 0.8))
    }

    @Test
    fun longReversalRescansSection() = assertCounts("re-scan") { seed ->
        // Walk ~9.6 m, back 6 m (tracks of those pallets die), then forward to the end.
        SyntheticLineScene(SimConfig(seed = seed), walk(12.0 to 0.8, 10.0 to -0.6, 30.0 to 0.8))
    }

    @Test
    fun endsHalfwayBack() = assertCounts("ends after walking back") { seed ->
        // Correct answer is the net number of pallets passed.
        SyntheticLineScene(SimConfig(seed = seed), walk(20.0 to 0.8, 5.0 to -0.8))
    }

    @Test
    fun fastWalkLowFrameRate() = assertCounts("fast") { seed ->
        SyntheticLineScene(SimConfig(seed = seed, fps = 6.0), walk(14.0 to 1.6))
    }

    @Test
    fun cameraJerks() = assertCounts("jerks") { seed ->
        // Sudden 0.35 m jumps between two detector frames (a hand jerk).
        SyntheticLineScene(
            SimConfig(seed = seed),
            walk(6.0 to 0.8, 0.1 to 3.5, 6.0 to 0.8, 0.1 to 3.5, 6.0 to 0.8, 0.1 to -3.5, 12.0 to 0.8),
        )
    }

    @Test
    fun oppositeImageDirection() = assertCounts("left-to-right") { seed ->
        SyntheticLineScene(SimConfig(seed = seed, imageDirection = 1))
    }

    @Test
    fun farRowSliversNeedTheSizeFilter() {
        val scene = { seed: Long -> SyntheticLineScene(SimConfig(seed = seed, farRowSlivers = true)) }
        // With the relative size filter enabled the far-row boxes are ignored.
        assertCounts("far row + size filter", PipelineConfig(counter = CounterConfig(relativeSizeFilter = 0.8f)), scene)
        // Without it they are (wrongly) counted — documents why the filter exists.
        val unfiltered = run(scene(1L))
        assertTrue(unfiltered > scene(1L).expectedCount(), "expected overcount without filter, got $unfiltered")
    }

    @Test
    fun expectedCountHelper() {
        assertEquals(16, SyntheticLineScene(SimConfig()).expectedCount())
        val back = SyntheticLineScene(SimConfig(), walk(20.0 to 0.8, 5.0 to -0.8))
        assertTrue(back.expectedCount() in 8..11)
    }
}
