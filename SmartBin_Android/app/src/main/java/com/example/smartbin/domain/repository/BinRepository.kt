package com.example.smartbin.domain.repository

import com.example.smartbin.domain.model.Bin
import com.example.smartbin.domain.model.WasteEvent
import java.time.Instant
import kotlinx.coroutines.flow.Flow

interface BinRepository {
    fun observeBins(): Flow<List<Bin>>
    fun observeBin(binId: String): Flow<Bin?>
    fun observeStreamStatus(): Flow<Boolean>
    fun getEvents(
        binIds: Set<String>,
        localities: Set<String>,
        startTime: Instant,
        endTime: Instant,
    ): Flow<List<WasteEvent>>

    fun streamEvents(): Flow<WasteEvent>
    suspend fun triggerDemoEvent(binId: String? = null)
}
