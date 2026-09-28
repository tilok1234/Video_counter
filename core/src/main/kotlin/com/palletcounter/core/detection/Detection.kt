package com.palletcounter.core.detection

import com.palletcounter.core.geometry.Box
import kotlinx.serialization.Serializable

/** Name of the single class the counting system cares about. */
const val EUR_PALLET_BASE = "eur_pallet_base"

/**
 * One detector output. [box] is in normalized upright-frame coordinates
 * (see [Box]); [classId] indexes the model's class list.
 */
@Serializable
data class Detection(
    val box: Box,
    val confidence: Float,
    val classId: Int = 0,
)

/**
 * All detections produced for one frame, stamped with the frame's monotonic timestamp.
 *
 * The timestamp is what the tracker uses for motion prediction, so live camera frames,
 * decoded video frames and replayed logs all flow through the same code path.
 */
@Serializable
data class DetectionFrame(
    val timestampNanos: Long,
    val detections: List<Detection>,
    /** Pixel size of the upright analysed frame (informational; 0 if unknown). */
    val frameWidth: Int = 0,
    val frameHeight: Int = 0,
    /** Detector latency for this frame in milliseconds, or -1 if unknown. */
    val inferenceMillis: Float = -1f,
    val frameIndex: Long = -1,
)

/** Static description of a detector implementation, shown in the debug overlay and logs. */
@Serializable
data class DetectorInfo(
    val name: String,
    /** e.g. "tflite-yolo", "simulated", "replay". */
    val kind: String,
    val inputWidth: Int = 0,
    val inputHeight: Int = 0,
    /** e.g. "GPU", "CPU x4", "n/a". */
    val accelerator: String = "n/a",
    /**
     * True for the plumbing-test detector that ignores image content. Such a detector must
     * never be presented as real computer vision; the UI shows a warning banner for it.
     */
    val isSimulation: Boolean = false,
    val details: Map<String, String> = emptyMap(),
)

/**
 * Platform-neutral detector contract.
 *
 * The Android app implements it for camera/video bitmaps (TFLite YOLO, simulation). Other
 * runtimes (ONNX, a different architecture) can be added without touching tracking,
 * counting or UI code, because everything downstream only sees [DetectionFrame]s.
 */
interface PalletDetector<in F> : AutoCloseable {
    val info: DetectorInfo

    /** Runs detection on [frame]. Implementations must be safe to call from one worker thread. */
    fun detect(frame: F, timestampNanos: Long): DetectionFrame

    override fun close() {}
}
