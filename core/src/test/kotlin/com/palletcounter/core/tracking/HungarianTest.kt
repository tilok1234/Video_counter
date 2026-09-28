package com.palletcounter.core.tracking

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class HungarianTest {
    @Test
    fun squareMatrixOptimalAssignment() {
        val cost = arrayOf(
            floatArrayOf(4f, 1f, 3f),
            floatArrayOf(2f, 0f, 5f),
            floatArrayOf(3f, 2f, 2f),
        )
        // optimum: (0,1)=1, (1,0)=2, (2,2)=2 -> 5
        val res = Hungarian.solve(cost).toSet()
        assertEquals(setOf(0 to 1, 1 to 0, 2 to 2), res)
    }

    @Test
    fun rectangularMoreColumns() {
        val cost = arrayOf(floatArrayOf(5f, 1f, 9f), floatArrayOf(1f, 5f, 9f))
        assertEquals(setOf(0 to 1, 1 to 0), Hungarian.solve(cost).toSet())
    }

    @Test
    fun rectangularMoreRows() {
        val cost = arrayOf(floatArrayOf(5f), floatArrayOf(1f), floatArrayOf(3f))
        assertEquals(listOf(1 to 0), Hungarian.solve(cost))
    }

    @Test
    fun infeasibleEntriesAreNeverMatched() {
        val inf = Hungarian.INFEASIBLE
        val cost = arrayOf(floatArrayOf(inf, inf), floatArrayOf(0.2f, inf))
        assertEquals(listOf(1 to 0), Hungarian.solve(cost))
    }

    @Test
    fun prefersMoreMatchesOverSingleCheapOne() {
        val inf = Hungarian.INFEASIBLE
        // Row 0 can take col 0 (0.1) or col 1 (0.5); row 1 can only take col 0 (0.4).
        val cost = arrayOf(floatArrayOf(0.1f, 0.5f), floatArrayOf(0.4f, inf))
        val res = Hungarian.solve(cost).toSet()
        assertTrue(res == setOf(0 to 1, 1 to 0), "got $res")
    }
}
