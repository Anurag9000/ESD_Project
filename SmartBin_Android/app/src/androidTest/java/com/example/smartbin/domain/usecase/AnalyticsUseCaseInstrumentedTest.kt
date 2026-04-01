package com.example.smartbin.domain.usecase

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.smartbin.data.repository.WasteClassConfigStore
import com.example.smartbin.domain.model.Bin
import com.example.smartbin.domain.model.BinStatus
import com.example.smartbin.domain.model.WasteEvent
import com.example.smartbin.domain.repository.BinRepository
import java.time.Instant
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.emptyFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class AnalyticsUseCaseInstrumentedTest {

    private lateinit var wasteClassConfigStore: WasteClassConfigStore

    @Before
    fun setUp() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        context.getSharedPreferences("smartbin_class_config", 0).edit().clear().commit()
        wasteClassConfigStore = WasteClassConfigStore(context)
        wasteClassConfigStore.saveConfiguration(
            classCount = 4,
            selectedPrimaryClasses = listOf("metal", "organic", "paper"),
        )
    }

    @Test
    fun analyticsCollapsesNonSelectedRawClassesIntoOther() = runBlocking {
        val repository = fakeRepository(
            bins = listOf(testBin(id = "B1", locality = "Central Plaza")),
            events = listOf(
                WasteEvent("E1", "B1", "metal", 0.95f, Instant.parse("2026-03-28T10:00:00Z")),
                WasteEvent("E2", "B1", "organic", 0.80f, Instant.parse("2026-03-28T12:00:00Z")),
                WasteEvent("E3", "B1", "glass", 0.82f, Instant.parse("2026-03-29T09:00:00Z")),
                WasteEvent("E4", "B1", "plastic", 0.77f, Instant.parse("2026-03-29T11:00:00Z")),
            ),
        )

        val result = GetAggregatedAnalyticsUseCase(repository, wasteClassConfigStore)(
            binIds = setOf("B1"),
            localities = emptySet(),
            startTime = Instant.parse("2026-03-01T00:00:00Z"),
            endTime = Instant.parse("2026-03-31T23:59:59Z"),
            bucket = AnalyticsBucket.DAY,
        ).first()

        assertEquals(listOf("Metal", "Organic", "Paper", "Other"), result.displayClasses)
        assertEquals(4, result.totalEvents)
        assertEquals(1, result.countsByLabel["Metal"])
        assertEquals(1, result.countsByLabel["Organic"])
        assertEquals(0, result.countsByLabel["Paper"])
        assertEquals(2, result.countsByLabel["Other"])
        assertEquals(0.5f, result.percentagesByLabel["Other"]!!, 0.001f)
        assertEquals(2, result.trend.size)
        assertEquals(1, result.trend[0].countsByLabel["Metal"])
        assertEquals(2, result.trend[1].countsByLabel["Other"])
    }

    private fun fakeRepository(
        bins: List<Bin>,
        events: List<WasteEvent>,
    ): BinRepository = object : BinRepository {
        override fun observeBins(): Flow<List<Bin>> = flowOf(bins)

        override fun observeBin(binId: String): Flow<Bin?> = flowOf(bins.find { it.id == binId })

        override fun observeBinsLoading(): Flow<Boolean> = flowOf(false)

        override fun observeRepositoryErrors(): Flow<String?> = flowOf(null)

        override fun observeStreamStatus(): Flow<Boolean> = flowOf(true)

        override fun getEvents(
            binIds: Set<String>,
            localities: Set<String>,
            startTime: Instant,
            endTime: Instant,
        ): Flow<List<WasteEvent>> {
            val matchedBinIds = when {
                binIds.isNotEmpty() -> binIds
                localities.isNotEmpty() -> bins.filter { it.locality in localities }.mapTo(linkedSetOf()) { it.id }
                else -> emptySet()
            }
            return flowOf(
                events.filter { event ->
                    val matchesSelection = when {
                        binIds.isNotEmpty() -> event.binId in binIds
                        localities.isNotEmpty() -> event.binId in matchedBinIds
                        else -> true
                    }
                    val matchesTime = !event.timestamp.isBefore(startTime) && !event.timestamp.isAfter(endTime)
                    matchesSelection && matchesTime
                },
            )
        }

        override fun streamEvents(): Flow<WasteEvent> = emptyFlow()

        override suspend fun triggerDemoEvent(binId: String?) = Unit
    }

    private fun testBin(id: String, locality: String): Bin = Bin(
        id = id,
        name = "Bin $id",
        latitude = 0.0,
        longitude = 0.0,
        locality = locality,
        status = BinStatus.ONLINE,
    )
}
