package com.example.smartbin.data.repository

import com.example.smartbin.BuildConfig
import com.example.smartbin.data.remote.BinApi
import com.example.smartbin.data.remote.CreateWasteEventRequest
import com.example.smartbin.data.remote.toDomain
import com.example.smartbin.domain.model.Bin
import com.example.smartbin.domain.model.BinStatus
import com.example.smartbin.domain.model.WasteEvent
import com.example.smartbin.domain.model.WasteType
import com.example.smartbin.domain.repository.BinRepository
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.random.Random
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import timber.log.Timber

@Singleton
class RemoteBinRepository @Inject constructor(
    private val binApi: BinApi,
    private val okHttpClient: OkHttpClient,
    private val json: kotlinx.serialization.json.Json,
) : BinRepository {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val binsState = MutableStateFlow<List<Bin>>(emptyList())
    private val streamStatus = MutableStateFlow(false)
    private val liveEvents = MutableSharedFlow<WasteEvent>(extraBufferCapacity = 64)
    private var webSocket: WebSocket? = null
    private var reconnectJob: Job? = null
    @Volatile
    private var closedByApp = false
    @Volatile
    private var started = false

    override fun observeBins(): Flow<List<Bin>> {
        ensureStarted()
        return binsState
    }

    override fun observeBin(binId: String): Flow<Bin?> {
        ensureStarted()
        return binsState.map { bins ->
            bins.find { it.id == binId }
        }
    }

    override fun observeStreamStatus(): Flow<Boolean> {
        ensureStarted()
        return streamStatus
    }

    override fun getEvents(
        binIds: Set<String>,
        localities: Set<String>,
        startTime: java.time.Instant,
        endTime: java.time.Instant,
    ): Flow<List<WasteEvent>> {
        ensureStarted()
        return flow {
            val events = binApi.getEvents(
                binIds = binIds.takeIf { it.isNotEmpty() }?.joinToString(","),
                localities = localities.takeIf { it.isNotEmpty() }?.joinToString(","),
                startTime = startTime.toString(),
                endTime = endTime.toString(),
            ).map { it.toDomain() }
            emit(events)
        }
    }

    override fun streamEvents(): Flow<WasteEvent> {
        ensureStarted()
        return liveEvents
    }

    override suspend fun triggerDemoEvent(binId: String?) {
        ensureStarted()
        val chosenBin = binId ?: binsState.value.firstOrNull { it.status != BinStatus.OFFLINE }?.id ?: return
        val request = CreateWasteEventRequest(
            binId = chosenBin,
            predictedClass = WasteType.entries.random(Random.Default).name.lowercase(),
            confidence = Random.Default.nextDouble(0.82, 0.99).toFloat(),
            eventTime = java.time.Instant.now().toString(),
            sourceDeviceId = "android-demo-client",
            modelVersion = "android-demo-trigger",
        )
        runCatching {
            binApi.postEvent(request).toDomain()
        }.onSuccess { event ->
            liveEvents.tryEmit(event)
            refreshBins()
        }.onFailure { error ->
            Timber.w(error, "Failed to post demo event to backend")
        }
    }

    @Synchronized
    private fun ensureStarted() {
        if (started) return
        started = true
        scope.launch { refreshBins() }
        connectStream()
    }

    private fun connectStream() {
        closedByApp = false
        val request = Request.Builder()
            .url(BuildConfig.WS_EVENTS_URL)
            .build()
        webSocket?.cancel()
        webSocket = okHttpClient.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                streamStatus.value = true
                reconnectJob?.cancel()
                scope.launch { refreshBins() }
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                scope.launch {
                    runCatching {
                        json.decodeFromString<com.example.smartbin.data.remote.WasteEventDto>(text).toDomain()
                    }.onSuccess { event ->
                        liveEvents.emit(event)
                        refreshBins()
                    }.onFailure { error ->
                        Timber.w(error, "Failed to decode waste event message")
                    }
                }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                streamStatus.value = false
                webSocket.close(code, reason)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                streamStatus.value = false
                scheduleReconnect()
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                streamStatus.value = false
                Timber.w(t, "Waste event stream failed")
                scheduleReconnect()
            }
        })
    }

    private fun scheduleReconnect() {
        if (closedByApp) return
        if (reconnectJob?.isActive == true) return
        reconnectJob = scope.launch {
            delay(3_000)
            connectStream()
        }
    }

    private suspend fun refreshBins() {
        runCatching {
            binApi.getBins().map { it.toDomain() }
        }.onSuccess { bins ->
            binsState.value = bins
        }.onFailure { error ->
            Timber.w(error, "Failed to refresh bins from backend")
        }
    }
}
