package com.example.smartbin.domain.usecase

import com.example.smartbin.domain.model.Bin
import com.example.smartbin.domain.model.BinStatus
import com.example.smartbin.domain.model.WasteEvent
import com.example.smartbin.domain.model.WasteType
import com.example.smartbin.domain.repository.BinRepository
import java.time.Instant
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.emptyFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Test

class AnalyticsUseCaseTest {

    @Test
    fun `when events are fetched by bin, aggregation is correct`() = runBlocking {
        val repository = fakeRepository(
            bins = listOf(
                testBin(id = "B1", locality = "Central Plaza"),
            ),
            events = listOf(
                WasteEvent("E1", "B1", WasteType.METAL, 0.9f, Instant.parse("2026-03-28T10:00:00Z")),
                WasteEvent("E2", "B1", WasteType.METAL, 0.8f, Instant.parse("2026-03-28T12:00:00Z")),
                WasteEvent("E3", "B1", WasteType.ORGANIC, 0.7f, Instant.parse("2026-03-29T09:00:00Z")),
            ),
        )

        val result = GetAggregatedAnalyticsUseCase(repository)(
            binIds = setOf("B1"),
            localities = emptySet(),
            startTime = Instant.parse("2026-03-01T00:00:00Z"),
            endTime = Instant.parse("2026-03-31T23:59:59Z"),
            bucket = AnalyticsBucket.DAY,
        ).first()

        assertEquals(3, result.totalEvents)
        assertEquals(1, result.selectedBinCount)
        assertEquals(0, result.selectedLocalityCount)
        assertEquals(2, result.countsByType[WasteType.METAL])
        assertEquals(1, result.countsByType[WasteType.ORGANIC])
        assertEquals(0.6666667f, result.percentagesByType[WasteType.METAL]!!, 0.01f)
        assertEquals(2, result.trend.size)
    }

    @Test
    fun `when localities are selected, analytics counts all bins in those localities`() = runBlocking {
        val repository = fakeRepository(
            bins = listOf(
                testBin(id = "B1", locality = "Central Plaza"),
                testBin(id = "B2", locality = "Central Plaza"),
                testBin(id = "B3", locality = "Food Court"),
            ),
            events = listOf(
                WasteEvent("E1", "B1", WasteType.METAL, 0.91f, Instant.parse("2026-03-28T10:00:00Z")),
                WasteEvent("E2", "B2", WasteType.PAPER, 0.83f, Instant.parse("2026-03-28T11:00:00Z")),
                WasteEvent("E3", "B3", WasteType.ORGANIC, 0.79f, Instant.parse("2026-03-28T12:00:00Z")),
            ),
        )

        val result = GetAggregatedAnalyticsUseCase(repository)(
            binIds = emptySet(),
            localities = setOf("Central Plaza"),
            startTime = Instant.parse("2026-03-01T00:00:00Z"),
            endTime = Instant.parse("2026-03-31T23:59:59Z"),
            bucket = AnalyticsBucket.DAY,
        ).first()

        assertEquals(2, result.totalEvents)
        assertEquals(2, result.selectedBinCount)
        assertEquals(1, result.selectedLocalityCount)
        assertEquals(1, result.countsByType[WasteType.METAL])
        assertEquals(1, result.countsByType[WasteType.PAPER])
        assertEquals(0, result.countsByType[WasteType.ORGANIC])
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
