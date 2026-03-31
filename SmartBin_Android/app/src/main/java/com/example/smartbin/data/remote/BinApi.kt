package com.example.smartbin.data.remote

import com.example.smartbin.domain.model.Bin
import com.example.smartbin.domain.model.BinStatus
import com.example.smartbin.domain.model.WasteEvent
import com.example.smartbin.domain.model.WasteType
import java.time.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Path
import retrofit2.http.Query

@Serializable
data class BinDto(
    @SerialName("bin_id") val binId: String,
    @SerialName("bin_name") val binName: String,
    val latitude: Double,
    val longitude: Double,
    val locality: String,
    val status: String,
    @SerialName("last_seen_at") val lastSeenAt: String? = null,
    @SerialName("installed_at") val installedAt: String? = null,
    @SerialName("last_waste_type") val lastWasteType: String? = null,
    @SerialName("total_events_today") val totalEventsToday: Int = 0,
)

@Serializable
data class WasteEventDto(
    @SerialName("event_id") val eventId: String? = null,
    @SerialName("bin_id") val binId: String,
    @SerialName("predicted_class") val predictedClass: String,
    val confidence: Float,
    @SerialName("event_time") val eventTime: String,
    @SerialName("uploaded_at") val uploadedAt: String? = null,
    @SerialName("source_device_id") val sourceDeviceId: String? = null,
    @SerialName("model_version") val modelVersion: String? = null,
)

@Serializable
data class CreateWasteEventRequest(
    @SerialName("bin_id") val binId: String,
    @SerialName("predicted_class") val predictedClass: String,
    val confidence: Float,
    @SerialName("event_time") val eventTime: String,
    @SerialName("source_device_id") val sourceDeviceId: String,
    @SerialName("model_version") val modelVersion: String,
)

interface BinApi {
    @GET("bins")
    suspend fun getBins(): List<BinDto>

    @GET("bins/{binId}")
    suspend fun getBin(@Path("binId") binId: String): BinDto

    @GET("events")
    suspend fun getEvents(
        @Query("bin_ids") binIds: String? = null,
        @Query("localities") localities: String? = null,
        @Query("start") startTime: String,
        @Query("end") endTime: String,
    ): List<WasteEventDto>

    @POST("events")
    suspend fun postEvent(@Body request: CreateWasteEventRequest): WasteEventDto
}

fun BinDto.toDomain(): Bin = Bin(
    id = binId,
    name = binName,
    latitude = latitude,
    longitude = longitude,
    locality = locality,
    status = when (status.lowercase()) {
        "online" -> BinStatus.ONLINE
        "offline" -> BinStatus.OFFLINE
        else -> BinStatus.DEGRADED
    },
    lastSeenAt = lastSeenAt?.let(Instant::parse),
    installedAt = installedAt?.let(Instant::parse),
    lastWasteType = lastWasteType?.let(WasteType::fromString),
    totalEventsToday = totalEventsToday,
)

fun WasteEventDto.toDomain(): WasteEvent = WasteEvent(
    id = eventId ?: "${binId}_${eventTime}",
    binId = binId,
    wasteType = WasteType.fromString(predictedClass),
    confidence = confidence,
    timestamp = Instant.parse(eventTime),
    uploadedAt = uploadedAt?.let(Instant::parse) ?: Instant.parse(eventTime),
    sourceDeviceId = sourceDeviceId ?: "unknown-source",
    modelVersion = modelVersion ?: "unknown-model",
)
