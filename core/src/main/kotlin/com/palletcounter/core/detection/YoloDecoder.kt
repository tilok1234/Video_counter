package com.palletcounter.core.detection

import kotlin.math.max
import kotlin.math.roundToInt

/** Memory layout of a YOLO-family detection output tensor. */
enum class YoloOutputFormat {
    /** Infer from the tensor shape and the model input size. */
    AUTO,

    /**
     * `[1, 4 + numClasses, numAnchors]`: rows are cx, cy, w, h followed by per-class scores
     * (already sigmoid-activated). Default for Ultralytics YOLOv8 / YOLO11 exports.
     */
    RAW_CHANNELS_FIRST,

    /** `[1, numAnchors, 4 + numClasses]`: transposed variant of [RAW_CHANNELS_FIRST]. */
    RAW_CHANNELS_LAST,

    /**
     * `[1, maxDetections, 6]`: x1, y1, x2, y2, score, classId. NMS-free end-to-end heads
     * (YOLOv10, YOLO26 default export).
     */
    END_TO_END,
}

/** Whether box coordinates in the output are relative to the input size or in input pixels. */
enum class CoordinateSpace { AUTO, NORMALIZED, PIXELS }

data class YoloDecoderConfig(
    val format: YoloOutputFormat = YoloOutputFormat.AUTO,
    val coordinates: CoordinateSpace = CoordinateSpace.AUTO,
    /** Class ids to keep; null keeps every class. */
    val targetClassIds: Set<Int>? = null,
    /**
     * Minimum score to emit a detection. Kept low on purpose: the tracker uses
     * low-confidence detections to keep existing tracks alive (ByteTrack).
     */
    val confidenceThreshold: Float = 0.1f,
    val iouThreshold: Float = 0.5f,
    val maxDetections: Int = 100,
    val classAgnosticNms: Boolean = true,
    /** End-to-end heads are trained to be duplicate-free; NMS is optional for them. */
    val applyNmsToEndToEnd: Boolean = false,
)

/** Concrete interpretation of an output tensor. */
data class ResolvedLayout(
    val format: YoloOutputFormat,
    val numAnchors: Int,
    val numChannels: Int,
) {
    val numClasses: Int get() = if (format == YoloOutputFormat.END_TO_END) -1 else numChannels - 4
}

object YoloLayout {
    private val DEFAULT_STRIDES = intArrayOf(8, 16, 32)

    /** Number of anchor points a stride-8/16/32 YOLO head produces for this input size. */
    fun expectedAnchorCount(inputWidth: Int, inputHeight: Int, strides: IntArray = DEFAULT_STRIDES): Int =
        strides.sumOf { s -> ((inputHeight + s - 1) / s) * ((inputWidth + s - 1) / s) }

    /**
     * Resolves [shape] (e.g. `[1, 5, 8400]`) into a [ResolvedLayout].
     *
     * @throws IllegalArgumentException if the shape cannot be a supported YOLO output.
     */
    fun resolve(
        shape: IntArray,
        inputWidth: Int,
        inputHeight: Int,
        requested: YoloOutputFormat = YoloOutputFormat.AUTO,
    ): ResolvedLayout {
        val dims = when (shape.size) {
            2 -> shape
            3 -> {
                require(shape[0] == 1) { "Only batch size 1 is supported, got ${shape.contentToString()}" }
                intArrayOf(shape[1], shape[2])
            }
            else -> throw IllegalArgumentException(
                "Unsupported YOLO output rank ${shape.size}: ${shape.contentToString()}",
            )
        }
        val a = dims[0]
        val b = dims[1]
        val layout = when (requested) {
            YoloOutputFormat.RAW_CHANNELS_FIRST -> ResolvedLayout(requested, numAnchors = b, numChannels = a)
            YoloOutputFormat.RAW_CHANNELS_LAST -> ResolvedLayout(requested, numAnchors = a, numChannels = b)
            YoloOutputFormat.END_TO_END -> ResolvedLayout(requested, numAnchors = a, numChannels = b)
            YoloOutputFormat.AUTO -> {
                val expected = expectedAnchorCount(inputWidth, inputHeight)
                when {
                    b == expected && a >= 5 -> ResolvedLayout(YoloOutputFormat.RAW_CHANNELS_FIRST, b, a)
                    a == expected && b >= 5 -> ResolvedLayout(YoloOutputFormat.RAW_CHANNELS_LAST, a, b)
                    b == 6 && a < expected -> ResolvedLayout(YoloOutputFormat.END_TO_END, a, b)
                    a in 5 until b -> ResolvedLayout(YoloOutputFormat.RAW_CHANNELS_FIRST, b, a)
                    b in 5 until a -> ResolvedLayout(YoloOutputFormat.RAW_CHANNELS_LAST, a, b)
                    else -> throw IllegalArgumentException(
                        "Cannot infer YOLO layout from ${shape.contentToString()} " +
                            "for input ${inputWidth}x$inputHeight; set the format in the model sidecar",
                    )
                }
            }
        }
        if (layout.format == YoloOutputFormat.END_TO_END) {
            require(layout.numChannels >= 6) { "End-to-end output needs >= 6 values per row" }
        } else {
            require(layout.numChannels >= 5) { "Raw output needs 4 box values + >= 1 class score" }
        }
        return layout
    }
}

/**
 * Converts a YOLO output tensor into [Detection]s in normalized upright-frame coordinates.
 *
 * Pure Kotlin so it can be unit-tested on the JVM against golden tensors exported from the
 * real model (see core/src/test/resources/golden).
 */
class YoloDecoder(val config: YoloDecoderConfig = YoloDecoderConfig()) {

    fun decode(output: FloatArray, shape: IntArray, letterbox: Letterbox): List<Detection> {
        val layout = YoloLayout.resolve(shape, letterbox.dstWidth, letterbox.dstHeight, config.format)
        require(output.size >= layout.numAnchors * layout.numChannels) {
            "Output buffer has ${output.size} values, layout needs ${layout.numAnchors * layout.numChannels}"
        }
        return decode(output, layout, letterbox)
    }

    fun decode(output: FloatArray, layout: ResolvedLayout, letterbox: Letterbox): List<Detection> =
        when (layout.format) {
            YoloOutputFormat.END_TO_END -> decodeEndToEnd(output, layout, letterbox)
            YoloOutputFormat.RAW_CHANNELS_FIRST, YoloOutputFormat.RAW_CHANNELS_LAST ->
                decodeRaw(output, layout, letterbox)
            YoloOutputFormat.AUTO -> error("layout must be resolved")
        }

    private class Candidate(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val score: Float,
        val classId: Int,
    )

    private fun decodeRaw(output: FloatArray, layout: ResolvedLayout, letterbox: Letterbox): List<Detection> {
        val n = layout.numAnchors
        val c = layout.numChannels
        val nc = c - 4
        val channelsFirst = layout.format == YoloOutputFormat.RAW_CHANNELS_FIRST
        val targets = config.targetClassIds
        val threshold = config.confidenceThreshold
        val candidates = ArrayList<Candidate>()
        var maxCoord = 0f
        for (i in 0 until n) {
            var best = -1f
            var bestClass = -1
            for (k in 0 until nc) {
                if (targets != null && k !in targets) continue
                val s = if (channelsFirst) output[(4 + k) * n + i] else output[i * c + 4 + k]
                if (s > best) {
                    best = s
                    bestClass = k
                }
            }
            if (bestClass < 0 || best < threshold) continue
            val cx: Float
            val cy: Float
            val w: Float
            val h: Float
            if (channelsFirst) {
                cx = output[i]; cy = output[n + i]; w = output[2 * n + i]; h = output[3 * n + i]
            } else {
                val base = i * c
                cx = output[base]; cy = output[base + 1]; w = output[base + 2]; h = output[base + 3]
            }
            maxCoord = max(maxCoord, max(max(cx, cy), max(w, h)))
            candidates += Candidate(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f, best, bestClass)
        }
        val detections = toDetections(candidates, maxCoord, letterbox)
        return nonMaxSuppression(
            detections,
            iouThreshold = config.iouThreshold,
            maxDetections = config.maxDetections,
            classAgnostic = config.classAgnosticNms,
        )
    }

    private fun decodeEndToEnd(output: FloatArray, layout: ResolvedLayout, letterbox: Letterbox): List<Detection> {
        val n = layout.numAnchors
        val c = layout.numChannels
        val targets = config.targetClassIds
        val candidates = ArrayList<Candidate>()
        var maxCoord = 0f
        for (i in 0 until n) {
            val base = i * c
            val score = output[base + 4]
            if (score < config.confidenceThreshold) continue
            val cls = output[base + 5].roundToInt()
            if (targets != null && cls !in targets) continue
            val x1 = output[base]
            val y1 = output[base + 1]
            val x2 = output[base + 2]
            val y2 = output[base + 3]
            maxCoord = max(maxCoord, max(max(x1, y1), max(x2, y2)))
            candidates += Candidate(x1, y1, x2, y2, score, cls)
        }
        val detections = toDetections(candidates, maxCoord, letterbox)
            .sortedByDescending { it.confidence }
        return if (config.applyNmsToEndToEnd) {
            nonMaxSuppression(detections, config.iouThreshold, config.maxDetections, config.classAgnosticNms)
        } else {
            detections.take(config.maxDetections)
        }
    }

    private fun toDetections(candidates: List<Candidate>, maxCoord: Float, letterbox: Letterbox): List<Detection> {
        if (candidates.isEmpty()) return emptyList()
        val normalized = when (config.coordinates) {
            CoordinateSpace.NORMALIZED -> true
            CoordinateSpace.PIXELS -> false
            // Normalized outputs never exceed ~1; pixel outputs of any real detection do.
            CoordinateSpace.AUTO -> maxCoord <= 2f
        }
        val sx = if (normalized) letterbox.dstWidth.toFloat() else 1f
        val sy = if (normalized) letterbox.dstHeight.toFloat() else 1f
        val result = ArrayList<Detection>(candidates.size)
        for (cand in candidates) {
            val box = letterbox.inputPixelsToSource(cand.x1 * sx, cand.y1 * sy, cand.x2 * sx, cand.y2 * sy)
                .clippedToUnit()
            if (!box.isValid || box.width < MIN_SIDE || box.height < MIN_SIDE) continue
            result += Detection(box, cand.score, cand.classId)
        }
        return result
    }

    private companion object {
        /** Boxes thinner than this (normalized) after clipping are discarded as degenerate. */
        const val MIN_SIDE = 1e-3f
    }
}
