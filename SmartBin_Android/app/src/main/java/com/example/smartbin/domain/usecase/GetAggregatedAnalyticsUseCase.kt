package com.example.smartbin.domain.usecase

import com.example.smartbin.domain.model.WasteEvent
import com.example.smartbin.domain.model.WasteType
import com.example.smartbin.domain.repository.BinRepository
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.map
import javax.inject.Inject

enum class AnalyticsBucket {
    DAY,
    WEEK,
    MONTH,
}

data class AnalyticsTrendPoint(
    val label: String,
    val totalEvents: Int,
    val countsByType: Map<WasteType, Int>,
)

data class AnalyticsResult(
    val totalEvents: Int,
    val countsByType: Map<WasteType, Int>,
    val percentagesByType: Map<WasteType, Float>,
    val averageConfidence: Float,
    val averageConfidenceByType: Map<WasteType, Float>,
    val trend: List<AnalyticsTrendPoint>,
    val selectedBinCount: Int,
    val selectedLocalityCount: Int,
)

class GetAggregatedAnalyticsUseCase @Inject constructor(
    private val binRepository: BinRepository,
) {
    operator fun invoke(
        binIds: Set<String>,
        localities: Set<String>,
        startTime: Instant,
        endTime: Instant,
        bucket: AnalyticsBucket,
    ): Flow<AnalyticsResult> {
        return combine(
            binRepository.observeBins(),
            binRepository.getEvents(
                binIds = binIds,
                localities = localities,
                startTime = startTime,
                endTime = endTime,
            ),
        ) { bins, events ->
            val selectedBinCount = when {
                binIds.isNotEmpty() -> bins.count { it.id in binIds }
                localities.isNotEmpty() -> bins.count { it.locality in localities }
                else -> 0
            }
            val total = events.size
            val counts = WasteType.entries.associateWith { type -> events.count { it.wasteType == type } }
            val percentages = counts.mapValues { (_, count) ->
                if (total > 0) count.toFloat() / total else 0f
            }
            val averageConfidence = if (events.isNotEmpty()) {
                events.map { it.confidence }.average().toFloat()
            } else {
                0f
            }
            val averageConfidenceByType = WasteType.entries.associateWith { type ->
                val typeEvents = events.filter { it.wasteType == type }
                if (typeEvents.isNotEmpty()) typeEvents.map { it.confidence }.average().toFloat() else 0f
            }

            AnalyticsResult(
                totalEvents = total,
                countsByType = counts,
                percentagesByType = percentages,
                averageConfidence = averageConfidence,
                averageConfidenceByType = averageConfidenceByType,
                trend = buildTrend(events, bucket),
                selectedBinCount = selectedBinCount,
                selectedLocalityCount = localities.size,
            )
        }
    }

    private fun buildTrend(events: List<WasteEvent>, bucket: AnalyticsBucket): List<AnalyticsTrendPoint> {
        if (events.isEmpty()) return emptyList()
        val zoneId = ZoneId.systemDefault()
        val formatter = when (bucket) {
            AnalyticsBucket.DAY -> DateTimeFormatter.ofPattern("dd MMM")
            AnalyticsBucket.WEEK -> DateTimeFormatter.ofPattern("'Wk' w")
            AnalyticsBucket.MONTH -> DateTimeFormatter.ofPattern("MMM")
        }

        return events
            .groupBy { event ->
                val localDate = event.timestamp.atZone(zoneId).toLocalDate()
                when (bucket) {
                    AnalyticsBucket.DAY -> localDate
                    AnalyticsBucket.WEEK -> localDate.minusDays((localDate.dayOfWeek.value - 1).toLong())
                    AnalyticsBucket.MONTH -> LocalDate.of(localDate.year, localDate.month, 1)
                }
            }
            .toSortedMap()
            .map { (bucketDate, bucketEvents) ->
                AnalyticsTrendPoint(
                    label = formatter.format(bucketDate),
                    totalEvents = bucketEvents.size,
                    countsByType = WasteType.entries.associateWith { type ->
                        bucketEvents.count { it.wasteType == type }
                    },
                )
            }
    }
}
