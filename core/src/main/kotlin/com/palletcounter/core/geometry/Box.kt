package com.palletcounter.core.geometry

import kotlinx.serialization.Serializable
import kotlin.math.max
import kotlin.math.min

/**
 * Axis-aligned rectangle in normalized coordinates of the *upright* analysed frame:
 * x grows to the right, y grows downward, (0,0) is the top-left corner and (1,1) the
 * bottom-right corner.
 *
 * All vision code (decoding, tracking, counting, overlays, logs) works in this space so
 * that it does not depend on camera resolution, rotation or model input size.
 */
@Serializable
data class Box(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
) {
    val width: Float get() = right - left
    val height: Float get() = bottom - top
    val centerX: Float get() = (left + right) * 0.5f
    val centerY: Float get() = (top + bottom) * 0.5f
    val area: Float get() = max(0f, width) * max(0f, height)

    val isValid: Boolean
        get() = left.isFinite() && top.isFinite() && right.isFinite() && bottom.isFinite() &&
            right > left && bottom > top

    fun intersectionArea(other: Box): Float {
        val w = min(right, other.right) - max(left, other.left)
        val h = min(bottom, other.bottom) - max(top, other.top)
        return if (w <= 0f || h <= 0f) 0f else w * h
    }

    /** Intersection over union in [0, 1]. */
    fun iou(other: Box): Float {
        val inter = intersectionArea(other)
        if (inter <= 0f) return 0f
        val union = area + other.area - inter
        return if (union <= 0f) 0f else inter / union
    }

    /**
     * Distance-IoU in (-1, 1]: IoU minus the squared centre distance divided by the squared
     * diagonal of the smallest enclosing box. Unlike IoU it still ranks non-overlapping
     * boxes by how close they are, which helps re-acquiring briefly lost tracks.
     */
    fun diou(other: Box): Float {
        val dx = centerX - other.centerX
        val dy = centerY - other.centerY
        val encW = max(right, other.right) - min(left, other.left)
        val encH = max(bottom, other.bottom) - min(top, other.top)
        val diag2 = encW * encW + encH * encH
        if (diag2 <= 0f) return iou(other)
        return iou(other) - (dx * dx + dy * dy) / diag2
    }

    /** Fraction of this box's area that lies inside [region]. */
    fun fractionInside(region: Box): Float {
        val a = area
        return if (a <= 0f) 0f else intersectionArea(region) / a
    }

    fun contains(x: Float, y: Float): Boolean = x >= left && x <= right && y >= top && y <= bottom

    fun translated(dx: Float, dy: Float): Box = Box(left + dx, top + dy, right + dx, bottom + dy)

    fun clippedToUnit(): Box = Box(
        left.coerceIn(0f, 1f),
        top.coerceIn(0f, 1f),
        right.coerceIn(0f, 1f),
        bottom.coerceIn(0f, 1f),
    )

    override fun toString(): String =
        "Box(%.3f, %.3f, %.3f, %.3f)".format(left, top, right, bottom)

    companion object {
        val UNIT = Box(0f, 0f, 1f, 1f)

        fun fromCenter(cx: Float, cy: Float, w: Float, h: Float): Box =
            Box(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f)
    }
}
