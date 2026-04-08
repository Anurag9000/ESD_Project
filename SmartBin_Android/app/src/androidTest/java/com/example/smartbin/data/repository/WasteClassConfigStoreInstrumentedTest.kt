package com.example.smartbin.data.repository

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assert.assertEquals

@RunWith(AndroidJUnit4::class)
class WasteClassConfigStoreInstrumentedTest {

    private lateinit var context: Context

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        context.getSharedPreferences("smartbin_class_config", Context.MODE_PRIVATE).edit().clear().commit()
    }

    @Test
    fun saveAndReloadPreservesSelectedPrimaryClassOrder() {
        val store = WasteClassConfigStore(context)
        store.saveConfiguration(
            classCount = 4,
            selectedPrimaryClasses = listOf("metal", "organic", "paper"),
            mergedAssignments = mapOf("glass" to "metal"),
        )

        val reloadedStore = WasteClassConfigStore(context)

        assertEquals(
            listOf("metal", "organic", "paper"),
            reloadedStore.resolvedConfiguration.value.selectedPrimaryClasses,
        )
        assertEquals(
            mapOf("glass" to "metal"),
            reloadedStore.resolvedConfiguration.value.mergedAssignments,
        )
        assertEquals(
            listOf("Metal", "Organic", "Paper", "Other"),
            reloadedStore.resolvedConfiguration.value.runtimeDisplayLabels,
        )
    }
}
