package com.example.smartbin.data.repository

import com.example.smartbin.data.remote.BinApi
import com.example.smartbin.data.remote.CreateWasteEventRequest
import com.example.smartbin.data.remote.liveServerHint
import com.example.smartbin.data.remote.resolvedWsEventsUrl
import com.example.smartbin.data.remote.toDomain
import com.example.smartbin.domain.model.Bin
import com.example.smartbin.domain.model.BinStatus
import com.example.smartbin.domain.model.WasteEvent
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
    private val wasteClassConfigStore: WasteClassConfigStore,
) : BinRepository {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val binsState = MutableStateFlow<List<Bin>>(emptyList())
    private val binsLoading = MutableStateFlow(true)
    private val repositoryErrors = MutableStateFlow<String?>(null)
    private val streamStatus = MutableStateFlow(false)
    private val liveEvents = MutableSharedFlow<WasteEvent>(extraBufferCapacity = 64)
    private val recentlyEmittedEventIds = LinkedHashSet<String>()
    private var webSocket: WebSocket? = null
    private var reconnectJob: Job? = null
    private var refreshBinsJob: Job? = null
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

    override fun observeBinsLoading(): Flow<Boolean> {
        ensureStarted()
        return binsLoading
    }

    override fun observeRepositoryErrors(): Flow<String?> {
        ensureStarted()
        return repositoryErrors
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
        val availableClasses = wasteClassConfigStore.catalog.value.availableRawClasses
        if (availableClasses.isEmpty()) return
        val chosenBin = binId ?: binsState.value.firstOrNull { it.status != BinStatus.OFFLINE }?.id ?: return
        val request = CreateWasteEventRequest(
            binId = chosenBin,
            predictedClass = availableClasses.random(Random.Default),
            confidence = Random.Default.nextDouble(0.82, 0.99).toFloat(),
            eventTime = java.time.Instant.now().toString(),
            sourceDeviceId = "android-demo-client",
            modelVersion = "android-demo-trigger",
        )
        runCatching {
            binApi.postEvent(request).toDomain()
        }.onSuccess { event ->
            repositoryErrors.value = null
            emitIfNew(event)
            applyLiveEventToBins(event)
            scheduleBinsRefresh()
        }.onFailure { error ->
            repositoryErrors.value = error.message ?: "Failed to post live demo event"
            Timber.w(error, "Failed to post demo event to backend")
        }
    }

    fun setActive(isActive: Boolean) {
        if (isActive) {
            if (started && webSocket == null) {
                binsLoading.value = true
                scope.launch { refreshBins() }
                connectStream()
            }
            return
        }
        closedByApp = true
        reconnectJob?.cancel()
        reconnectJob = null
        refreshBinsJob?.cancel()
        refreshBinsJob = null
        webSocket?.close(1000, "Switched away from live mode")
        webSocket = null
        streamStatus.value = false
        repositoryErrors.value = null
    }

    @Synchronized
    private fun ensureStarted() {
        if (started) return
        started = true
        binsLoading.value = true
        scope.launch { refreshBins() }
        connectStream()
    }

    private fun connectStream() {
        closedByApp = false
        val request = Request.Builder()
            .url(resolvedWsEventsUrl())
            .build()
        webSocket?.cancel()
        webSocket = okHttpClient.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                streamStatus.value = true
                repositoryErrors.value = null
                reconnectJob?.cancel()
                scheduleBinsRefresh(immediate = true)
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                scope.launch {
                    runCatching {
                        json.decodeFromString<com.example.smartbin.data.remote.WasteEventDto>(text).toDomain()
                    }.onSuccess { event ->
                        emitIfNew(event)
                        repositoryErrors.value = null
                        applyLiveEventToBins(event)
                        scheduleBinsRefresh()
                    }.onFailure { error ->
                        repositoryErrors.value = error.message ?: "Failed to decode live event"
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
                repositoryErrors.value = userFriendlyLiveError(t)
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

    private fun scheduleBinsRefresh(immediate: Boolean = false) {
        refreshBinsJob?.cancel()
        refreshBinsJob = scope.launch {
            if (!immediate) {
                delay(350)
            }
            refreshBins()
        }
    }

    private suspend fun refreshBins() {
        binsLoading.value = true
        runCatching {
            binApi.getBins().map { it.toDomain() }
        }.onSuccess { bins ->
            binsState.value = bins
            binsLoading.value = false
            repositoryErrors.value = null
        }.onFailure { error ->
            binsLoading.value = false
            repositoryErrors.value = userFriendlyLiveError(error)
            Timber.w(error, "Failed to refresh bins from backend")
        }
    }

    private fun userFriendlyLiveError(error: Throwable): String {
        val message = error.message.orEmpty()
        return when {
            message.contains("CLEARTEXT communication", ignoreCase = true) ->
                "Live server blocked by Android network policy. ${liveServerHint()}"
            message.contains("Failed to connect", ignoreCase = true) ||
                message.contains("Connection refused", ignoreCase = true) ->
                "Live server unavailable. ${liveServerHint()}"
            else -> message.ifBlank { "Live server unavailable. ${liveServerHint()}" }
        }
    }

    private fun applyLiveEventToBins(event: WasteEvent) {
        binsState.update { bins ->
            bins.map { bin ->
                if (bin.id != event.binId) return@map bin
                bin.copy(
                    status = BinStatus.ONLINE,
                    lastSeenAt = event.timestamp,
                    lastPredictedClass = event.predictedClass,
                    totalEventsToday = bin.totalEventsToday + 1,
                )
            }
        }
    }

    private suspend fun emitIfNew(event: WasteEvent) {
        val wasAdded = synchronized(recentlyEmittedEventIds) {
            if (!recentlyEmittedEventIds.add(event.id)) {
                false
            } else {
                while (recentlyEmittedEventIds.size > 128) {
                    val oldest = recentlyEmittedEventIds.firstOrNull() ?: break
                    recentlyEmittedEventIds.remove(oldest)
                }
                true
            }
        }
        if (wasAdded) {
            liveEvents.emit(event)
        }
    }
}
