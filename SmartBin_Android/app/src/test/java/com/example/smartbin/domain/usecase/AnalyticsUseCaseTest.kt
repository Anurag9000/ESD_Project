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
    fun `when events are fetched, aggregation is correct`() = runBlocking {
        val repository = object : BinRepository {
            override fun observeBins(): Flow<List<Bin>> = flowOf(emptyList())

            override fun observeBin(binId: String): Flow<Bin?> = flowOf(
                Bin(
                    id = "B1",
                    name = "Test",
                    latitude = 0.0,
                    longitude = 0.0,
                    locality = "Locality",
                    status = BinStatus.ONLINE,
                ),
            )

            override fun observeStreamStatus(): Flow<Boolean> = flowOf(true)

            override fun getEvents(
                binIds: Set<String>,
                localities: Set<String>,
                startTime: Instant,
                endTime: Instant,
            ): Flow<List<WasteEvent>> = flowOf(
                listOf(
                    WasteEvent("E1", "B1", WasteType.METAL, 0.9f, Instant.parse("2026-03-28T10:00:00Z")),
                    WasteEvent("E2", "B1", WasteType.METAL, 0.8f, Instant.parse("2026-03-28T12:00:00Z")),
                    WasteEvent("E3", "B1", WasteType.ORGANIC, 0.7f, Instant.parse("2026-03-29T09:00:00Z")),
                ),
            )

            override fun streamEvents(): Flow<WasteEvent> = emptyFlow()

            override suspend fun triggerDemoEvent(binId: String?) = Unit
        }

        val result = GetAggregatedAnalyticsUseCase(repository)(
            binIds = setOf("B1"),
            localities = emptySet(),
            startTime = Instant.parse("2026-03-01T00:00:00Z"),
            endTime = Instant.parse("2026-03-31T23:59:59Z"),
            bucket = AnalyticsBucket.DAY,
        ).first()

        assertEquals(3, result.totalEvents)
        assertEquals(2, result.countsByType[WasteType.METAL])
        assertEquals(1, result.countsByType[WasteType.ORGANIC])
        assertEquals(0.6666667f, result.percentagesByType[WasteType.METAL]!!, 0.01f)
        assertEquals(2, result.trend.size)
    }
}
