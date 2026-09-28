package com.palletcounter.app.camera

import android.graphics.Bitmap
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.palletcounter.app.detection.FrameInput
import java.nio.ByteBuffer

/**
 * Receives every camera frame (for the camera-FPS readout) but converts and forwards a
 * frame only when [gate] says the detector is ready for one. The RGBA buffer is copied into
 * one reused bitmap, so no per-frame allocations happen; the gate guarantees the previous
 * frame has been fully processed before the bitmap is overwritten.
 */
class FrameAnalyzer(
    private val gate: FrameGate,
) : ImageAnalysis.Analyzer {
    private var bitmap: Bitmap? = null
    private var packed: ByteBuffer? = null

    override fun analyze(image: ImageProxy) {
        val timestamp = image.imageInfo.timestamp
        gate.onCameraFrame(timestamp)
        if (!gate.tryBegin(timestamp)) {
            image.close()
            return
        }
        try {
            val w = image.width
            val h = image.height
            val bmp = bitmap?.takeIf { it.width == w && it.height == h }
                ?: Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also { bitmap = it }
            copyRgba(image, bmp)
            val frame = FrameInput(bmp, w, h, image.imageInfo.rotationDegrees, timestamp)
            image.close()
            gate.submit(frame)
        } catch (t: Throwable) {
            image.close()
            gate.abort(t)
        }
    }

    /**
     * Copies the RGBA_8888 plane into [bmp]. Rows can be padded (rowStride > width * 4) and
     * the last row is often not padded, so padded buffers are packed row by row first.
     */
    private fun copyRgba(image: ImageProxy, bmp: Bitmap) {
        val plane = image.planes[0]
        val src = plane.buffer
        val rowBytes = image.width * 4
        src.rewind()
        if (plane.rowStride == rowBytes && plane.pixelStride == 4) {
            bmp.copyPixelsFromBuffer(src)
            return
        }
        val dst = packed?.takeIf { it.capacity() == rowBytes * image.height }
            ?: ByteBuffer.allocateDirect(rowBytes * image.height).also { packed = it }
        dst.clear()
        val row = ByteArray(rowBytes)
        for (y in 0 until image.height) {
            src.position(y * plane.rowStride)
            src.get(row, 0, rowBytes)
            dst.put(row)
        }
        dst.flip()
        bmp.copyPixelsFromBuffer(dst)
    }
}

/** Implemented by the scan engine: decides which frames are processed. */
interface FrameGate {
    fun onCameraFrame(timestampNanos: Long)

    /** Returns true (and reserves the worker) if a new frame should be converted now. */
    fun tryBegin(timestampNanos: Long): Boolean

    /** Hands over a frame after a successful [tryBegin]. */
    fun submit(frame: FrameInput)

    /** Releases a reservation made by [tryBegin] when conversion failed. */
    fun abort(error: Throwable)
}
