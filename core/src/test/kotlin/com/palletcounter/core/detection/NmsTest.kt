package com.palletcounter.core.detection

import com.palletcounter.core.geometry.Box
import kotlin.test.Test
import kotlin.test.assertEquals

class NmsTest {
    @Test
    fun suppressesOverlappingLowerScoredBox() {
        val a = Detection(Box(0.1f, 0.1f, 0.4f, 0.3f), 0.9f)
        val b = Detection(Box(0.11f, 0.1f, 0.41f, 0.3f), 0.8f)
        val c = Detection(Box(0.6f, 0.1f, 0.9f, 0.3f), 0.7f)
        val kept = nonMaxSuppression(listOf(b, c, a), iouThreshold = 0.5f)
        assertEquals(listOf(a, c), kept)
    }

    @Test
    fun keepsAdjacentTouchingPallets() {
        // Two pallets side by side touch but barely overlap: both must survive.
        val a = Detection(Box(0.10f, 0.5f, 0.40f, 0.6f), 0.9f)
        val b = Detection(Box(0.39f, 0.5f, 0.69f, 0.6f), 0.85f)
        assertEquals(2, nonMaxSuppression(listOf(a, b), iouThreshold = 0.5f).size)
    }

    @Test
    fun perClassNmsKeepsDifferentClasses() {
        val a = Detection(Box(0.1f, 0.1f, 0.4f, 0.3f), 0.9f, classId = 0)
        val b = Detection(Box(0.1f, 0.1f, 0.4f, 0.3f), 0.8f, classId = 1)
        assertEquals(2, nonMaxSuppression(listOf(a, b), 0.5f, classAgnostic = false).size)
        assertEquals(1, nonMaxSuppression(listOf(a, b), 0.5f, classAgnostic = true).size)
    }
}
