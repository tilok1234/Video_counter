package com.palletcounter.core.session

import kotlinx.serialization.Serializable
import kotlin.math.max

/** Half-pallets stacked on one EUR-pallet in a line. Chosen by the user, never inferred. */
@Serializable
enum class StackSize(val halfPallets: Int) {
    TWENTY(20),
    THIRTY(30),
}

/** How the scanned count relates to the whole line. */
@Serializable
enum class LineMode(val multiplier: Int, val label: String) {
    /** Only the visible side of a two-wide line was scanned: multiply by 2. */
    ONE_SIDE_X2(2, "ONE SIDE ×2"),

    /** Every pallet of the line was scanned: multiply by 1. */
    WHOLE_LINE_X1(1, "WHOLE LINE ×1"),
}

@Serializable
data class ScanSettings(
    val stackSize: StackSize = StackSize.THIRTY,
    val lineMode: LineMode = LineMode.ONE_SIDE_X2,
)

/**
 * Detected count plus manual correction, and the derived totals shown on the review screen.
 *
 * Example: 16 pallets on one side, stack size 30, one side ×2 → 16 × 30 × 2 = 960.
 */
@Serializable
data class CountResult(
    val detectedPallets: Int,
    val manualAdjustment: Int = 0,
    val settings: ScanSettings = ScanSettings(),
) {
    /** EUR-pallets counted on the scanned side after manual correction (never negative). */
    val pallets: Int get() = max(0, detectedPallets + manualAdjustment)

    /** EUR-pallets in the whole line. */
    val palletsInLine: Int get() = pallets * settings.lineMode.multiplier

    val halfPallets: Int get() = pallets * settings.stackSize.halfPallets * settings.lineMode.multiplier

    val formula: String
        get() = "$pallets × ${settings.stackSize.halfPallets} × ${settings.lineMode.multiplier} = $halfPallets"

    fun adjusted(delta: Int): CountResult {
        // Clamp so the corrected count cannot go below zero.
        val newAdjustment = max(-detectedPallets, manualAdjustment + delta)
        return copy(manualAdjustment = newAdjustment)
    }

    fun withSettings(newSettings: ScanSettings): CountResult = copy(settings = newSettings)
}
