package com.palletcounter.core.session

import kotlin.test.Test
import kotlin.test.assertEquals

class CountResultTest {
    @Test
    fun exampleFromRequirements() {
        val r = CountResult(16, settings = ScanSettings(StackSize.THIRTY, LineMode.ONE_SIDE_X2))
        assertEquals(960, r.halfPallets)
        assertEquals(32, r.palletsInLine)
        assertEquals("16 × 30 × 2 = 960", r.formula)
    }

    @Test
    fun reviewScreenExample() {
        val r = CountResult(17, settings = ScanSettings(StackSize.THIRTY, LineMode.ONE_SIDE_X2))
        assertEquals(1020, r.halfPallets)
    }

    @Test
    fun manualCorrectionUpdatesTotalAndNeverGoesNegative() {
        var r = CountResult(2, settings = ScanSettings(StackSize.TWENTY, LineMode.WHOLE_LINE_X1))
        r = r.adjusted(+1)
        assertEquals(3, r.pallets)
        assertEquals(60, r.halfPallets)
        repeat(10) { r = r.adjusted(-1) }
        assertEquals(0, r.pallets)
        assertEquals(0, r.halfPallets)
        r = r.adjusted(+1)
        assertEquals(1, r.pallets)
    }

    @Test
    fun changingSettingsRecalculates() {
        val r = CountResult(10).withSettings(ScanSettings(StackSize.TWENTY, LineMode.WHOLE_LINE_X1))
        assertEquals(200, r.halfPallets)
    }
}
