package com.example.smartbin.domain.repository

import com.example.smartbin.domain.model.Bin
import com.example.smartbin.domain.model.WasteEvent
import kotlinx.coroutines.flow.Flow

interface BinRepository {
    fun getBins(): Flow<List<Bin>>
    fun getBin(binId: String): Flow<Bin?>
    fun getEvents(binIds: List<String>, startTime: String, endTime: String): Flow<List<WasteEvent>>
    fun streamEvents(): Flow<WasteEvent>
}
