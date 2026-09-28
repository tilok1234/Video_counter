package com.palletcounter.core.detection

import com.palletcounter.core.geometry.Box
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Aspect-preserving resize of an upright source image into a fixed model input, centred
 * with constant padding (the same "letterbox" Ultralytics uses at inference time).
 *
 * The Android preprocessor draws the frame using [scaledWidth]/[scaledHeight]/[padLeft]/[padTop]
 * and the decoder maps boxes back with [inputPixelsToSource], so both sides share one
 * definition of the geometry.
 */
data class Letterbox(
    val srcWidth: Int,
    val srcHeight: Int,
    val dstWidth: Int,
    val dstHeight: Int,
) {
    init {
        require(srcWidth > 0 && srcHeight > 0) { "source size must be positive" }
        require(dstWidth > 0 && dstHeight > 0) { "model input size must be positive" }
    }

    val scale: Float = min(dstWidth.toFloat() / srcWidth, dstHeight.toFloat() / srcHeight)
    val scaledWidth: Int = (srcWidth * scale).roundToInt().coerceIn(1, dstWidth)
    val scaledHeight: Int = (srcHeight * scale).roundToInt().coerceIn(1, dstHeight)
    val padLeft: Int = (dstWidth - scaledWidth) / 2
    val padTop: Int = (dstHeight - scaledHeight) / 2

    /** Maps a box given in model-input pixels to normalized source coordinates. */
    fun inputPixelsToSource(left: Float, top: Float, right: Float, bottom: Float): Box = Box(
        (left - padLeft) / scaledWidth,
        (top - padTop) / scaledHeight,
        (right - padLeft) / scaledWidth,
        (bottom - padTop) / scaledHeight,
    )

    /** Maps a normalized source box to model-input pixels (used by tests and tools). */
    fun sourceToInputPixels(box: Box): Box = Box(
        box.left * scaledWidth + padLeft,
        box.top * scaledHeight + padTop,
        box.right * scaledWidth + padLeft,
        box.bottom * scaledHeight + padTop,
    )
}
