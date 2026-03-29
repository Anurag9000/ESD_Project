package com.example.smartbin.domain.usecase

import com.example.smartbin.domain.model.WasteEvent
import com.example.smartbin.domain.model.WasteType
import com.example.smartbin.domain.repository.BinRepository
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.mockito.Mock
import org.mockito.Mockito.`when`
import org.mockito.MockitoAnnotations

class AnalyticsUseCaseTest {

    @Mock
    lateinit var repository: BinRepository

    private lateinit var useCase: GetAggregatedAnalyticsUseCase

    @Before
    fun setup() {
        MockitoAnnotations.openMocks(this)
        useCase = GetAggregatedAnalyticsUseCase(repository)
    }

    @Test
    fun `when events are fetched, aggregation is correct`() = runTest {
        // Arrange
        val events = listOf(
            WasteEvent("B1", WasteType.METAL, 0.9f, "2026-03-28"),
            WasteEvent("B1", WasteType.METAL, 0.8f, "2026-03-28"),
            WasteEvent("B1", WasteType.ORGANIC, 0.7f, "2026-03-28")
        )
        `when`(repository.getEvents(listOf("B1"), "start", "end")).thenReturn(flowOf(events))

        // Act
        val result = useCase(listOf("B1"), "start", "end").first()

        // Assert
        assertEquals(3, result.totalEvents)
        assertEquals(2, result.countsByType[WasteType.METAL])
        assertEquals(1, result.countsByType[WasteType.ORGANIC])
        assertEquals(0.6666667f, result.percentagesByType[WasteType.METAL]!!, 0.01f)
    }
}
