package com.palletcounter.app.detection

import com.palletcounter.core.detection.DetectionFrame
import com.palletcounter.core.detection.DetectorInfo
import com.palletcounter.core.detection.PalletDetector
import com.palletcounter.core.sim.Segment
import com.palletcounter.core.sim.SimConfig
import com.palletcounter.core.sim.SyntheticLineScene
import kotlin.random.Random

/**
 * PLUMBING TEST ONLY — NOT COMPUTER VISION.
 *
 * Ignores the camera image and emits synthetic boxes for an endless line of pallets
 * passing by at walking speed, so camera → tracking → counting → UI can be exercised before
 * a trained model exists. The UI shows a red "SIMULATION" banner whenever it is active and
 * results are flagged as simulated in history and logs.
 */
class SimulatedDetector : PalletDetector<FrameInput> {
    private val scene = SyntheticLineScene(
        SimConfig(endless = true, missProbability = 0.1, lowConfidenceProbability = 0.1, falsePositivesPerFrame = 0.05),
        listOf(Segment(durationS = 1.0, speedMps = 0.7)),
    )
    private val rng = Random(42)
    private var startNanos: Long? = null

    override val info = DetectorInfo(
        name = "SIMULATION (no model)",
        kind = "simulated",
        accelerator = "n/a",
        isSimulation = true,
    )

    override fun detect(frame: FrameInput, timestampNanos: Long): DetectionFrame {
        val start = startNanos ?: timestampNanos.also { startNanos = it }
        val t = (timestampNanos - start) / 1e9
        return DetectionFrame(
            timestampNanos = timestampNanos,
            detections = scene.detectionsAt(t, rng),
            frameWidth = frame.uprightWidth,
            frameHeight = frame.uprightHeight,
            inferenceMillis = 0f,
        )
    }
}
