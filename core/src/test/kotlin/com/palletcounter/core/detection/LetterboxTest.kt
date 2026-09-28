package com.palletcounter.core.detection

import com.palletcounter.core.geometry.Box
import kotlin.test.Test
import kotlin.test.assertEquals

class LetterboxTest {
    @Test
    fun landscapeFrameIntoSquareInputIsPaddedVertically() {
        val lb = Letterbox(srcWidth = 1280, srcHeight = 960, dstWidth = 640, dstHeight = 640)
        assertEquals(0.5f, lb.scale, 1e-6f)
        assertEquals(640, lb.scaledWidth)
        assertEquals(480, lb.scaledHeight)
        assertEquals(0, lb.padLeft)
        assertEquals(80, lb.padTop)
    }

    @Test
    fun portraitFrameIsPaddedHorizontally() {
        val lb = Letterbox(srcWidth = 960, srcHeight = 1280, dstWidth = 640, dstHeight = 640)
        assertEquals(480, lb.scaledWidth)
        assertEquals(80, lb.padLeft)
        assertEquals(0, lb.padTop)
    }

    @Test
    fun roundTripSourceToInputAndBack() {
        val lb = Letterbox(1920, 1080, 640, 640)
        val src = Box(0.25f, 0.4f, 0.6f, 0.9f)
        val px = lb.sourceToInputPixels(src)
        val back = lb.inputPixelsToSource(px.left, px.top, px.right, px.bottom)
        assertEquals(src.left, back.left, 1e-5f)
        assertEquals(src.top, back.top, 1e-5f)
        assertEquals(src.right, back.right, 1e-5f)
        assertEquals(src.bottom, back.bottom, 1e-5f)
    }
}
