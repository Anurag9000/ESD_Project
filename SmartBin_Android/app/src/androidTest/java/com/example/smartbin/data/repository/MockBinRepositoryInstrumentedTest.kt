package com.example.smartbin.data.repository

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.time.Instant
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class MockBinRepositoryInstrumentedTest {

    private lateinit var wasteClassConfigStore: WasteClassConfigStore

    @Before
    fun setUp() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        context.getSharedPreferences("smartbin_class_config", 0).edit().clear().commit()
        wasteClassConfigStore = WasteClassConfigStore(context)
    }

    @Test
    fun triggerDemoEventUpdatesEventStreamAndBinSnapshot() = runBlocking {
        val repository = MockBinRepository(wasteClassConfigStore)
        val targetBinId = "BIN-001"
        val initialEvents = repository.getEvents(
            binIds = setOf(targetBinId),
            localities = emptySet(),
            startTime = Instant.parse("2025-01-01T00:00:00Z"),
            endTime = Instant.parse("2027-01-01T00:00:00Z"),
        ).first()

        repository.triggerDemoEvent(targetBinId)

        val updatedEvents = repository.getEvents(
            binIds = setOf(targetBinId),
            localities = emptySet(),
            startTime = Instant.parse("2025-01-01T00:00:00Z"),
            endTime = Instant.parse("2027-01-01T00:00:00Z"),
        ).first()
        val updatedBin = repository.observeBin(targetBinId).first()

        assertEquals(initialEvents.size + 1, updatedEvents.size)
        assertNotNull(updatedBin)
        assertNotNull(updatedBin?.lastPredictedClass)
        assertTrue(updatedBin!!.totalEventsToday >= 1)
        assertTrue(updatedBin.lastPredictedClass in wasteClassConfigStore.catalog.value.availableRawClasses)
    }
}
