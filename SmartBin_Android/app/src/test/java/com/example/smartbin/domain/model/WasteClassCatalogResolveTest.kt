package com.example.smartbin.domain.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class WasteClassCatalogResolveTest {

    private val catalog = WasteClassCatalog(
        availableRawClasses = listOf("battery", "glass", "metal", "organic", "paper", "plastic"),
        recommendedPrimaryClasses = listOf("metal", "organic", "paper"),
        otherLabel = "Other",
    )

    @Test
    fun `default configuration resolves to three primary classes plus other`() {
        val resolved = catalog.resolve(WasteClassConfiguration())

        assertEquals(4, resolved.classCount)
        assertEquals(listOf("metal", "organic", "paper"), resolved.selectedPrimaryClasses)
        assertEquals(listOf("Metal", "Organic", "Paper", "Other"), resolved.runtimeDisplayLabels)
        assertEquals(listOf("battery", "glass", "plastic"), resolved.remainingRawClasses())
    }

    @Test
    fun `invalid selections are removed and class count is clamped`() {
        val resolved = catalog.resolve(
            WasteClassConfiguration(
                classCount = 99,
                selectedPrimaryClasses = listOf("glass", "glass", "missing", "battery"),
                userConfirmed = true,
            ),
        )

        assertEquals(catalog.availableRawClasses.size, resolved.classCount)
        assertEquals(listOf("glass", "battery", "metal", "organic", "paper"), resolved.selectedPrimaryClasses)
        assertFalse(resolved.remainingRawClasses().contains("glass"))
        assertEquals("Other", resolved.toRuntimeDisplayLabel("plastic"))
        assertEquals("Glass", resolved.toRuntimeDisplayLabel("glass"))
    }

    @Test
    fun `format waste class label normalizes separators and capitalization`() {
        assertEquals("E Waste", formatWasteClassLabel("e_waste"))
        assertEquals("Paper Cup", formatWasteClassLabel("paper-cup"))
        assertEquals("Mixed Plastic", formatWasteClassLabel("  mixed   plastic "))
    }
}
