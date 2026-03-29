package com.example.smartbin.data.remote

import com.example.smartbin.domain.model.Bin
import com.example.smartbin.domain.model.WasteEvent
import kotlinx.serialization.Serializable
import retrofit2.http.GET
import retrofit2.http.Path
import retrofit2.http.Query

@Serializable
data class BinDto(
    val bin_id: String,
    val bin_name: String,
    val latitude: Double,
    val longitude: Double,
    val locality: String,
    val status: String
)

@Serializable
data class WasteEventDto(
    val bin_id: String,
    val predicted_class: String,
    val confidence: Float,
    val event_time: String
)

interface BinApi {
    @GET("bins")
    suspend fun getBins(): List<BinDto>

    @GET("bins/{binId}")
    suspend fun getBin(@Path("binId") binId: String): BinDto

    @GET("events")
    suspend fun getEvents(
        @Query("bin_ids") binIds: String,
        @Query("start") startTime: String,
        @Query("end") endTime: String
    ): List<WasteEventDto>
}
