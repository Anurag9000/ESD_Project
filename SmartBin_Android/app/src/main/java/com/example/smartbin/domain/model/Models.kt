package com.example.smartbin.domain.model

import java.time.Instant

enum class WasteType(val label: String) {
    METAL("Metal"),
    ORGANIC("Organic"),
    PAPER("Paper"),
    OTHER("Other");

    companion object {
        fun fromString(value: String): WasteType {
            return entries.find { it.name.equals(value, ignoreCase = true) } ?: OTHER
        }
    }
}

enum class BinStatus(val label: String) {
    ONLINE("Online"),
    OFFLINE("Offline"),
    DEGRADED("Degraded");
}

data class Bin(
    val id: String,
    val name: String,
    val latitude: Double,
    val longitude: Double,
    val locality: String,
    val status: BinStatus,
    val lastSeenAt: Instant? = null,
    val installedAt: Instant? = null,
    val lastWasteType: WasteType? = null,
    val totalEventsToday: Int = 0,
)

data class WasteEvent(
    val id: String,
    val binId: String,
    val wasteType: WasteType,
    val confidence: Float,
    val timestamp: Instant,
    val uploadedAt: Instant = timestamp,
    val sourceDeviceId: String = "raspberry-pi-4",
    val modelVersion: String = "efficientnet-b0-ce-v1",
)

data class TimeRange(
    val start: Instant,
    val end: Instant,
)
