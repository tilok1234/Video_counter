package com.palletcounter.app.detection

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import com.palletcounter.core.detection.Letterbox
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

/** Element type of the model input tensor. */
enum class InputType { FLOAT32, UINT8, INT8 }

/**
 * Rotates + letterboxes a [FrameInput] into the model input tensor in one bitmap draw and
 * one pixel copy. Buffers are allocated once and reused for every frame.
 */
class Preprocessor(
    val inputWidth: Int,
    val inputHeight: Int,
    private val type: InputType,
    private val channelsFirst: Boolean,
    padValue: Int,
    private val normalize: Boolean,
    private val quantScale: Float = 1f,
    private val quantZeroPoint: Int = 0,
) {
    private val canvasBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(canvasBitmap)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val matrix = Matrix()
    private val pixels = IntArray(inputWidth * inputHeight)
    private val padColor = Color.rgb(padValue, padValue, padValue)
    private val bytesPerChannel = if (type == InputType.FLOAT32) 4 else 1
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3 * bytesPerChannel)
        .order(ByteOrder.nativeOrder())

    fun letterboxFor(frame: FrameInput) = Letterbox(frame.uprightWidth, frame.uprightHeight, inputWidth, inputHeight)

    fun process(frame: FrameInput, lb: Letterbox): ByteBuffer {
        canvas.drawColor(padColor)
        matrix.reset()
        matrix.postTranslate(-frame.width / 2f, -frame.height / 2f)
        matrix.postRotate(frame.rotationDegrees.toFloat())
        matrix.postScale(lb.scaledWidth / frame.uprightWidth.toFloat(), lb.scaledHeight / frame.uprightHeight.toFloat())
        matrix.postTranslate(lb.padLeft + lb.scaledWidth / 2f, lb.padTop + lb.scaledHeight / 2f)
        canvas.save()
        canvas.clipRect(lb.padLeft, lb.padTop, lb.padLeft + lb.scaledWidth, lb.padTop + lb.scaledHeight)
        canvas.drawBitmap(frame.bitmap, matrix, paint)
        canvas.restore()
        canvasBitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        fill()
        return buffer
    }

    private fun fill() {
        buffer.rewind()
        val n = pixels.size
        if (channelsFirst) {
            for (c in 0 until 3) {
                val shift = 16 - 8 * c // R, G, B
                for (i in 0 until n) put((pixels[i] shr shift) and 0xFF)
            }
        } else {
            for (i in 0 until n) {
                val p = pixels[i]
                put((p shr 16) and 0xFF)
                put((p shr 8) and 0xFF)
                put(p and 0xFF)
            }
        }
        buffer.rewind()
    }

    private fun put(v: Int) {
        when (type) {
            InputType.FLOAT32 -> buffer.putFloat(if (normalize) v / 255f else v.toFloat())
            InputType.UINT8 -> buffer.put(v.toByte())
            InputType.INT8 -> {
                val real = if (normalize) v / 255f else v.toFloat()
                val q = (real / quantScale + quantZeroPoint).roundToInt().coerceIn(-128, 127)
                buffer.put(q.toByte())
            }
        }
    }
}
