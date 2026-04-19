package com.example.smartbin.domain.usecase

import com.example.smartbin.data.repository.WasteClassConfigStore
import com.example.smartbin.domain.model.WasteEvent
import com.example.smartbin.domain.repository.BinRepository
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import javax.inject.Inject
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.combine

enum class AnalyticsBucket {
    DAY,
    WEEK,
    MONTH,
}

data class AnalyticsTrendPoint(
    val label: String,
    val totalEvents: Int,
    val countsByLabel: Map<String, Int>,
)

data class AnalyticsResult(
    val totalEvents: Int,
    val displayClasses: List<String>,
    val countsByLabel: Map<String, Int>,
    val percentagesByLabel: Map<String, Float>,
    val averageConfidence: Float,
    val averageConfidenceByLabel: Map<String, Float>,
    val trend: List<AnalyticsTrendPoint>,
    val selectedBinCount: Int,
    val selectedLocalityCount: Int,
)

class GetAggregatedAnalyticsUseCase @Inject constructor(
    private val binRepository: BinRepository,
    private val wasteClassConfigStore: WasteClassConfigStore,
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
            wasteClassConfigStore.resolvedConfiguration,
        ) { bins, events, resolvedConfiguration ->
            val selectedBinCount = when {
                binIds.isNotEmpty() -> bins.count { it.id in binIds }
                localities.isNotEmpty() -> bins.count { it.locality in localities }
                else -> 0
            }
            val displayClasses = resolvedConfiguration.runtimeDisplayLabels
            val total = events.size
            val groupedByDisplay = events.groupBy { event ->
                resolvedConfiguration.toRuntimeDisplayLabel(event.predictedClass)
            }
            val counts = displayClasses.associateWith { label -> groupedByDisplay[label].orEmpty().size }
            val percentages = counts.mapValues { (_, count) ->
                if (total > 0) count.toFloat() / total else 0f
            }
            val averageConfidence = if (events.isNotEmpty()) {
                events.map { it.confidence }.average().toFloat()
            } else {
                0f
            }
            val averageConfidenceByLabel = displayClasses.associateWith { label ->
                val labelEvents = groupedByDisplay[label].orEmpty()
                if (labelEvents.isNotEmpty()) labelEvents.map { it.confidence }.average().toFloat() else 0f
            }

            AnalyticsResult(
                totalEvents = total,
                displayClasses = displayClasses,
                countsByLabel = counts,
                percentagesByLabel = percentages,
                averageConfidence = averageConfidence,
                averageConfidenceByLabel = averageConfidenceByLabel,
                trend = buildTrend(events, displayClasses, resolvedConfiguration::toRuntimeDisplayLabel, bucket),
                selectedBinCount = selectedBinCount,
                selectedLocalityCount = localities.size,
            )
        }
    }

    private fun buildTrend(
        events: List<WasteEvent>,
        displayClasses: List<String>,
        displayMapper: (String?) -> String,
        bucket: AnalyticsBucket,
    ): List<AnalyticsTrendPoint> {
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
                val grouped = bucketEvents.groupBy { displayMapper(it.predictedClass) }
                AnalyticsTrendPoint(
                    label = formatter.format(bucketDate),
                    totalEvents = bucketEvents.size,
                    countsByLabel = displayClasses.associateWith { label -> grouped[label].orEmpty().size },
                )
            }
    }
}
