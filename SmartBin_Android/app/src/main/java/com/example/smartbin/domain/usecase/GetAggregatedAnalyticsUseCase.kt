package com.example.smartbin.domain.usecase

import com.example.smartbin.domain.model.WasteEvent
import com.example.smartbin.domain.model.WasteType
import com.example.smartbin.domain.repository.BinRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import javax.inject.Inject

data class AnalyticsResult(
    val totalEvents: Int,
    val countsByType: Map<WasteType, Int>,
    val percentagesByType: Map<WasteType, Float>
)

class GetAggregatedAnalyticsUseCase @Inject constructor(
    private val binRepository: BinRepository
) {
    operator fun invoke(
        binIds: List<String>,
        startTime: String,
        endTime: String
    ): Flow<AnalyticsResult> {
        return binRepository.getEvents(binIds, startTime, endTime).map { events ->
            val total = events.size
            val counts = events.groupBy { it.wasteType }.mapValues { it.value.size }
            val percentages = counts.mapValues { if (total > 0) it.value.toFloat() / total else 0f }

            AnalyticsResult(
                totalEvents = total,
                countsByType = counts,
                percentagesByType = percentages
            )
        }
    }
}
