package com.example.smartbin.domain.model

enum class WasteType {
    METAL, ORGANIC, PAPER, OTHER;

    companion object {
        fun fromString(value: String): WasteType {
            return entries.find { it.name.lowercase() == value.lowercase() } ?: OTHER
        }
    }
}

data class Bin(
    val id: String,
    val name: String,
    val latitude: Double,
    val longitude: Double,
    val locality: String,
    val status: String,
    val lastSeenAt: String? = null
)

data class WasteEvent(
    val binId: String,
    val wasteType: WasteType,
    val confidence: Float,
    val timestamp: String
)
