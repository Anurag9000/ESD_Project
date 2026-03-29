package com.example.smartbin.data.repository

import com.example.smartbin.domain.model.Bin
import com.example.smartbin.domain.model.WasteEvent
import com.example.smartbin.domain.model.WasteType
import com.example.smartbin.domain.repository.BinRepository
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.onStart
import java.time.Instant
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.random.Random

@Singleton
class MockBinRepository @Inject constructor() : BinRepository {

    private val mockBins = listOf(
        Bin("BIN-001", "Main Entrance Bin", 40.7128, -74.0060, "Downtown", "online"),
        Bin("BIN-002", "North Gate Bin", 40.7148, -74.0040, "North Park", "online"),
        Bin("BIN-003", "South Gate Bin", 40.7108, -74.0080, "South Park", "offline"),
        Bin("BIN-004", "Food Court Bin", 40.7138, -74.0070, "Market Center", "online")
    )

    private val eventFlow = MutableSharedFlow<WasteEvent>()

    /**
     * Call this from the UI (e.g., via a hidden button or ViewModel action)
     * to verify the live-demo real-time feedback loop.
     */
    suspend fun triggerManualEvent(binId: String, type: WasteType) {
        val event = WasteEvent(
            binId = binId,
            wasteType = type,
            confidence = 0.99f,
            timestamp = Instant.now().toString()
        )
        eventFlow.emit(event)
    }

    override fun getBins(): Flow<List<Bin>> = flow {
        emit(mockBins)
    }

    override fun getBin(binId: String): Flow<Bin?> = flow {
        emit(mockBins.find { it.id == binId })
    }

    override fun getEvents(binIds: List<String>, startTime: String, endTime: String): Flow<List<WasteEvent>> = flow {
        val randomEvents = List(20) {
            WasteEvent(
                binId = binIds.random(),
                wasteType = WasteType.entries.random(),
                confidence = Random.nextFloat(),
                timestamp = Instant.now().toString()
            )
        }
        emit(randomEvents)
    }

    override fun streamEvents(): Flow<WasteEvent> = eventFlow.onStart {
        // Optional: Emit initial events for background noise during demo
        repeat(10) {
            delay(10000) // Emit background noise every 10s
            val event = WasteEvent(
                binId = mockBins.random().id,
                wasteType = WasteType.entries.random(),
                confidence = Random.nextFloat(),
                timestamp = Instant.now().toString()
            )
            emit(event)
        }
    }
}
