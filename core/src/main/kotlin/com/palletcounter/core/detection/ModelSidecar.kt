package com.palletcounter.core.detection

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * Optional JSON file stored next to a `.tflite` model (same base name, `.json` extension).
 * Written by `training/export_model.py`; every field has a safe default so hand-written or
 * partial files work too. Tensor shapes/types read from the interpreter always win over
 * [inputWidth]/[inputHeight], which are informational.
 */
@Serializable
data class ModelSidecar(
    val name: String = "unnamed-model",
    val version: String = "",
    /** "auto", "yolo_raw", "yolo_raw_channels_last" or "yolo_end2end". */
    @SerialName("output_format") val outputFormat: String = "auto",
    /** "auto", "normalized" or "pixels". */
    val coordinates: String = "auto",
    @SerialName("input_width") val inputWidth: Int = 0,
    @SerialName("input_height") val inputHeight: Int = 0,
    /** "zero_one" (pixel / 255) or "none" (0..255 floats). */
    @SerialName("input_normalization") val inputNormalization: String = "zero_one",
    @SerialName("letterbox_pad_value") val letterboxPadValue: Int = 114,
    val classes: List<String> = listOf(EUR_PALLET_BASE),
    @SerialName("target_classes") val targetClasses: List<String> = listOf(EUR_PALLET_BASE),
    /** Suggested operating point from validation; the tracker thresholds are separate. */
    @SerialName("confidence_threshold") val confidenceThreshold: Float? = null,
    @SerialName("iou_threshold") val iouThreshold: Float? = null,
    val framework: String? = null,
    @SerialName("source_weights") val sourceWeights: String? = null,
    @SerialName("created_at") val createdAt: String? = null,
    val quantization: String? = null,
    val metrics: Map<String, Double> = emptyMap(),
    val notes: String? = null,
) {
    fun decoderFormat(): YoloOutputFormat = when (outputFormat.lowercase()) {
        "yolo_raw", "raw", "raw_channels_first" -> YoloOutputFormat.RAW_CHANNELS_FIRST
        "yolo_raw_channels_last", "raw_channels_last" -> YoloOutputFormat.RAW_CHANNELS_LAST
        "yolo_end2end", "end2end", "end_to_end" -> YoloOutputFormat.END_TO_END
        else -> YoloOutputFormat.AUTO
    }

    fun coordinateSpace(): CoordinateSpace = when (coordinates.lowercase()) {
        "normalized" -> CoordinateSpace.NORMALIZED
        "pixels" -> CoordinateSpace.PIXELS
        else -> CoordinateSpace.AUTO
    }

    /**
     * Class ids to count. Falls back to class 0 for single-class models, and to all classes
     * if none of the target names appear in [classes].
     */
    fun targetClassIds(): Set<Int>? {
        val ids = targetClasses.mapNotNull { name -> classes.indexOf(name).takeIf { it >= 0 } }.toSet()
        return when {
            ids.isNotEmpty() -> ids
            classes.size <= 1 -> setOf(0)
            else -> null
        }
    }

    fun toDecoderConfig(confidenceThreshold: Float, iouThreshold: Float? = null): YoloDecoderConfig =
        YoloDecoderConfig(
            format = decoderFormat(),
            coordinates = coordinateSpace(),
            targetClassIds = targetClassIds(),
            confidenceThreshold = confidenceThreshold,
            iouThreshold = iouThreshold ?: this.iouThreshold ?: 0.5f,
        )

    companion object {
        private val json = Json { ignoreUnknownKeys = true; isLenient = true }

        fun parse(text: String): ModelSidecar = json.decodeFromString(serializer(), text)

        fun toJson(sidecar: ModelSidecar): String =
            Json { prettyPrint = true; encodeDefaults = true }.encodeToString(serializer(), sidecar)
    }
}
