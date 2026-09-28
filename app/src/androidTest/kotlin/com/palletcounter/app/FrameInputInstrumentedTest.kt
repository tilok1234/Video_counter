package com.palletcounter.app

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.palletcounter.app.detection.FrameInput
import com.palletcounter.core.geometry.Box
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/** Upright-box → sensor-pixel mapping used for thumbnails and captures. */
@RunWith(AndroidJUnit4::class)
class FrameInputInstrumentedTest {
    private val bmp = Bitmap.createBitmap(400, 300, Bitmap.Config.ARGB_8888)

    @Test
    fun noRotationMapsDirectly() {
        val r = FrameInput(bmp, 400, 300, 0, 0).sourceRect(Box(0.25f, 0.5f, 0.5f, 1f))
        assertEquals(100, r.left); assertEquals(150, r.top); assertEquals(200, r.right); assertEquals(300, r.bottom)
    }

    @Test
    fun rotation90MapsUprightTopLeftToSensorBottomLeft() {
        // Upright frame is 300 x 400 (portrait). Its top-left quadrant comes from the sensor's
        // bottom-left quadrant, since rotating the sensor image 90° clockwise makes it upright.
        val f = FrameInput(bmp, 400, 300, 90, 0)
        assertEquals(300, f.uprightWidth)
        assertEquals(400, f.uprightHeight)
        val r = f.sourceRect(Box(0f, 0f, 0.5f, 0.5f))
        assertEquals(0, r.left); assertEquals(150, r.top); assertEquals(200, r.right); assertEquals(300, r.bottom)
    }

    @Test
    fun uprightCropHasUprightOrientation() {
        val src = Bitmap.createBitmap(400, 300, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.GREEN) }
        val crop = FrameInput(src, 400, 300, 90, 0).uprightCrop()
        assertEquals(300, crop.width)
        assertEquals(400, crop.height)
    }
}
