package com.example.smartbin.data.repository

import com.example.smartbin.domain.model.Bin
import com.example.smartbin.domain.model.BinStatus
import com.example.smartbin.domain.model.WasteEvent
import com.example.smartbin.domain.model.WasteType
import com.example.smartbin.domain.repository.BinRepository
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.temporal.ChronoUnit
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.random.Random
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

@Singleton
class MockBinRepository @Inject constructor() : BinRepository {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val random = Random(42)
    private val zoneId = ZoneId.systemDefault()

    private val baseBins = listOf(
        Bin(
            id = "BIN-001",
            name = "Campus Gate Alpha",
            latitude = 28.6139,
            longitude = 77.2090,
            locality = "Central Plaza",
            status = BinStatus.ONLINE,
            installedAt = Instant.parse("2026-01-05T08:00:00Z"),
        ),
        Bin(
            id = "BIN-002",
            name = "Library Walk",
            latitude = 28.6152,
            longitude = 77.2108,
            locality = "Academic Block",
            status = BinStatus.ONLINE,
            installedAt = Instant.parse("2026-01-08T08:00:00Z"),
        ),
        Bin(
            id = "BIN-003",
            name = "Food Court South",
            latitude = 28.6118,
            longitude = 77.2069,
            locality = "Food Court",
            status = BinStatus.DEGRADED,
            installedAt = Instant.parse("2026-01-11T08:00:00Z"),
        ),
        Bin(
            id = "BIN-004",
            name = "Hostel Entrance",
            latitude = 28.6174,
            longitude = 77.2144,
            locality = "Residential Zone",
            status = BinStatus.ONLINE,
            installedAt = Instant.parse("2026-01-15T08:00:00Z"),
        ),
        Bin(
            id = "BIN-005",
            name = "Market Corner",
            latitude = 28.6104,
            longitude = 77.2123,
            locality = "Market Street",
            status = BinStatus.OFFLINE,
            installedAt = Instant.parse("2026-01-19T08:00:00Z"),
        ),
        Bin(
            id = "BIN-006",
            name = "Research Annex",
            latitude = 28.6164,
            longitude = 77.2057,
            locality = "Innovation Park",
            status = BinStatus.ONLINE,
            installedAt = Instant.parse("2026-01-22T08:00:00Z"),
        ),
    )

    private val binsState = MutableStateFlow(baseBins)
    private val eventsState = MutableStateFlow(seedEvents())
    private val liveEvents = MutableSharedFlow<WasteEvent>(extraBufferCapacity = 32)
    private val streamStatus = MutableStateFlow(true)

    init {
        rebuildBinSnapshots()
        startBackgroundEvents()
    }

    override fun observeBins(): Flow<List<Bin>> = binsState

    override fun observeBin(binId: String): Flow<Bin?> = binsState.map { bins ->
        bins.find { it.id == binId }
    }

    override fun observeStreamStatus(): Flow<Boolean> = streamStatus

    override fun getEvents(
        binIds: Set<String>,
        localities: Set<String>,
        startTime: Instant,
        endTime: Instant,
    ): Flow<List<WasteEvent>> {
        return eventsState.map { events ->
            val matchedBinIds = if (localities.isEmpty()) {
                emptySet()
            } else {
                binsState.value.filter { it.locality in localities }.mapTo(mutableSetOf()) { it.id }
            }
            events.filter { event ->
                val matchesBin = when {
                    binIds.isNotEmpty() -> event.binId in binIds
                    localities.isNotEmpty() -> event.binId in matchedBinIds
                    else -> true
                }
                val matchesLocality = localities.isEmpty() || event.binId in matchedBinIds
                val matchesTime = !event.timestamp.isBefore(startTime) && !event.timestamp.isAfter(endTime)
                matchesBin && matchesLocality && matchesTime
            }
        }
    }

    override fun streamEvents(): Flow<WasteEvent> = liveEvents

    override suspend fun triggerDemoEvent(binId: String?) {
        emitNewEvent(
            binId = binId ?: binsState.value.filter { it.status != BinStatus.OFFLINE }.random(random).id,
            type = WasteType.entries.random(random),
            confidence = random.nextDouble(0.82, 0.99).toFloat(),
        )
    }

    private fun startBackgroundEvents() {
        scope.launch {
            while (true) {
                delay(9000)
                val candidateBins = binsState.value.filter { it.status != BinStatus.OFFLINE }
                if (candidateBins.isEmpty()) continue
                val bin = candidateBins.random(random)
                emitNewEvent(
                    binId = bin.id,
                    type = weightedWasteTypeFor(bin.locality),
                    confidence = random.nextDouble(0.72, 0.98).toFloat(),
                )
            }
        }
    }

    private suspend fun emitNewEvent(binId: String, type: WasteType, confidence: Float) {
        val now = Instant.now()
        val event = WasteEvent(
            id = "EVT-${now.toEpochMilli()}-$binId",
            binId = binId,
            wasteType = type,
            confidence = confidence,
            timestamp = now,
        )
        eventsState.update { it + event }
        rebuildBinSnapshots()
        liveEvents.emit(event)
    }

    private fun rebuildBinSnapshots() {
        val today = LocalDate.now(zoneId)
        val eventsByBin = eventsState.value.groupBy { it.binId }
        binsState.update { bins ->
            bins.map { bin ->
                val binEvents = eventsByBin[bin.id].orEmpty().sortedByDescending { it.timestamp }
                val lastEvent = binEvents.firstOrNull()
                val todayCount = binEvents.count {
                    it.timestamp.atZone(zoneId).toLocalDate() == today
                }
                val liveStatus = when {
                    bin.status == BinStatus.OFFLINE -> BinStatus.OFFLINE
                    lastEvent != null && ChronoUnit.MINUTES.between(lastEvent.timestamp, Instant.now()) <= 20 -> BinStatus.ONLINE
                    lastEvent != null && ChronoUnit.HOURS.between(lastEvent.timestamp, Instant.now()) <= 24 -> BinStatus.DEGRADED
                    else -> bin.status
                }
                bin.copy(
                    status = liveStatus,
                    lastSeenAt = lastEvent?.timestamp,
                    lastWasteType = lastEvent?.wasteType,
                    totalEventsToday = todayCount,
                )
            }
        }
    }

    private fun seedEvents(): List<WasteEvent> {
        val start = LocalDate.now(zoneId).minusDays(180)
        val events = mutableListOf<WasteEvent>()
        var eventCounter = 0

        generateSequence(start) { current ->
            if (current.isBefore(LocalDate.now(zoneId))) current.plusDays(1) else null
        }.take(181).forEach { day ->
            baseBins.forEach { bin ->
                val eventsForDay = when (bin.locality) {
                    "Food Court" -> random.nextInt(8, 16)
                    "Market Street" -> random.nextInt(6, 13)
                    "Residential Zone" -> random.nextInt(4, 10)
                    else -> random.nextInt(3, 9)
                }
                repeat(eventsForDay) {
                    val hour = random.nextInt(7, 22)
                    val minute = random.nextInt(0, 60)
                    val timestamp = day.atTime(hour, minute).atZone(zoneId).toInstant()
                    events += WasteEvent(
                        id = "EVT-${eventCounter++}",
                        binId = bin.id,
                        wasteType = weightedWasteTypeFor(bin.locality),
                        confidence = random.nextDouble(0.61, 0.98).toFloat(),
                        timestamp = timestamp,
                    )
                }
            }
        }

        return events.sortedBy { it.timestamp }
    }

    private fun weightedWasteTypeFor(locality: String): WasteType {
        val roll = random.nextInt(100)
        return when (locality) {
            "Food Court" -> when {
                roll < 58 -> WasteType.ORGANIC
                roll < 73 -> WasteType.PAPER
                roll < 88 -> WasteType.OTHER
                else -> WasteType.METAL
            }

            "Market Street" -> when {
                roll < 18 -> WasteType.ORGANIC
                roll < 42 -> WasteType.PAPER
                roll < 75 -> WasteType.OTHER
                else -> WasteType.METAL
            }

            "Residential Zone" -> when {
                roll < 33 -> WasteType.ORGANIC
                roll < 55 -> WasteType.PAPER
                roll < 83 -> WasteType.OTHER
                else -> WasteType.METAL
            }

            else -> when {
                roll < 22 -> WasteType.ORGANIC
                roll < 43 -> WasteType.PAPER
                roll < 76 -> WasteType.OTHER
                else -> WasteType.METAL
            }
        }
    }
}
