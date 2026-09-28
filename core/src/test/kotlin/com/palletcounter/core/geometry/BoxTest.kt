package com.palletcounter.core.geometry

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class BoxTest {
    @Test
    fun iouOfIdenticalBoxesIsOne() {
        val b = Box(0.1f, 0.2f, 0.4f, 0.5f)
        assertEquals(1f, b.iou(b), 1e-6f)
    }

    @Test
    fun iouOfDisjointBoxesIsZero() {
        assertEquals(0f, Box(0f, 0f, 0.1f, 0.1f).iou(Box(0.5f, 0.5f, 0.6f, 0.6f)))
    }

    @Test
    fun iouOfHalfOverlap() {
        val a = Box(0f, 0f, 0.2f, 0.1f)
        val b = Box(0.1f, 0f, 0.3f, 0.1f)
        // intersection 0.1*0.1, union 2*0.02 - 0.01
        assertEquals(0.01f / 0.03f, a.iou(b), 1e-5f)
    }

    @Test
    fun diouRanksNearNonOverlappingBoxesAboveFarOnes() {
        val a = Box(0f, 0f, 0.1f, 0.1f)
        val near = a.translated(0.12f, 0f)
        val far = a.translated(0.5f, 0f)
        assertEquals(0f, a.iou(near))
        assertTrue(a.diou(near) > a.diou(far))
        assertTrue(a.diou(near) < 0f)
    }

    @Test
    fun fractionInsideRegion() {
        val b = Box(0f, 0f, 0.2f, 0.2f)
        assertEquals(0.5f, b.fractionInside(Box(0f, 0.1f, 1f, 1f)), 1e-6f)
    }
}
