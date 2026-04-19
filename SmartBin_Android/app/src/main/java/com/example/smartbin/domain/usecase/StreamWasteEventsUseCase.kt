package com.example.smartbin.domain.usecase

import com.example.smartbin.domain.model.WasteEvent
import com.example.smartbin.domain.repository.BinRepository
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject

class StreamWasteEventsUseCase @Inject constructor(
    private val binRepository: BinRepository
) {
    operator fun invoke(): Flow<WasteEvent> {
        return binRepository.streamEvents()
    }
}
