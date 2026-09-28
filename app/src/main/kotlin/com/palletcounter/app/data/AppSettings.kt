package com.palletcounter.app.data

import com.palletcounter.core.counting.CounterConfig
import com.palletcounter.core.geometry.Box
import com.palletcounter.core.pipeline.PipelineConfig
import com.palletcounter.core.session.LineMode
import com.palletcounter.core.session.ScanSettings
import com.palletcounter.core.session.StackSize
import com.palletcounter.core.tracking.TrackerConfig
import kotlinx.serialization.Serializable

@Serializable
enum class DetectorMode { MODEL, SIMULATION }

@Serializable
enum class Accelerator(val label: String) { AUTO("Auto (GPU if supported)"), CPU("CPU"), GPU("GPU") }

@Serializable
enum class AnalysisResolution(val width: Int, val height: Int, val label: String) {
    LOW(640, 480, "640×480 (fast)"),
    HIGH(1280, 960, "1280×960 (sharper captures)"),
}

/** Region-of-interest presets; the guide band shown while scanning. */
@Serializable
enum class RoiPreset(val label: String, val top: Float, val bottom: Float) {
    CENTER_BAND("Center band", 0.25f, 0.85f),
    LOW_BAND("Low band", 0.45f, 0.97f),
    FULL_HEIGHT("Full height (all tiers)", 0.0f, 1.0f),
    CUSTOM("Custom", 0.25f, 0.85f),
}

/** Every user-adjustable option, persisted as one JSON document. */
@Serializable
data class AppSettings(
    val stackSize: StackSize = StackSize.THIRTY,
    val lineMode: LineMode = LineMode.ONE_SIDE_X2,
    val detectorMode: DetectorMode = DetectorMode.MODEL,
    val accelerator: Accelerator = Accelerator.AUTO,
    val cpuThreads: Int = 4,
    val maxInferenceFps: Int = 10,
    val analysisResolution: AnalysisResolution = AnalysisResolution.HIGH,
    val roiPreset: RoiPreset = RoiPreset.CENTER_BAND,
    val customRoiTop: Float = 0.25f,
    val customRoiBottom: Float = 0.85f,
    val lineX: Float = 0.5f,
    val hysteresis: Float = 0.04f,
    val highThreshold: Float = 0.5f,
    val lowThreshold: Float = 0.1f,
    val newTrackThreshold: Float = 0.55f,
    val minHits: Int = 3,
    val sizeFilter: Float = 0f,
    val showDebugOverlay: Boolean = true,
    val showRawDetections: Boolean = false,
    val showTrails: Boolean = false,
    val showCaptureButtons: Boolean = true,
    val recordDetectionLogs: Boolean = true,
    val photoConfidence: Float = 0.4f,
) {
    val scanSettings: ScanSettings get() = ScanSettings(stackSize, lineMode)

    fun roi(): Box {
        val (top, bottom) = if (roiPreset == RoiPreset.CUSTOM) customRoiTop to customRoiBottom else roiPreset.top to roiPreset.bottom
        return Box(0f, top.coerceIn(0f, 0.95f), 1f, bottom.coerceIn(top + 0.05f, 1f))
    }

    fun pipelineConfig(): PipelineConfig = PipelineConfig(
        roi = roi(),
        tracker = TrackerConfig(
            highThreshold = highThreshold,
            lowThreshold = lowThreshold,
            newTrackThreshold = maxOf(newTrackThreshold, highThreshold),
            minHitsToConfirm = minHits,
        ),
        counter = CounterConfig(
            lineX = lineX,
            hysteresis = hysteresis,
            minHits = minHits,
            relativeSizeFilter = sizeFilter,
        ),
    )
}
