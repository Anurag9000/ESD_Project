package com.example.smartbin.data.repository

import com.example.smartbin.domain.model.Bin
import com.example.smartbin.domain.model.WasteEvent
import com.example.smartbin.domain.repository.BinRepository
import java.time.Instant
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flatMapLatest

@Singleton
@OptIn(ExperimentalCoroutinesApi::class)
class HybridBinRepository @Inject constructor(
    private val mockRepository: MockBinRepository,
    private val remoteRepository: RemoteBinRepository,
    private val demoModeStore: DemoModeStore,
) : BinRepository {

    override fun observeBins(): Flow<List<Bin>> = demoModeStore.mode.flatMapLatest { mode ->
        active(mode).observeBins()
    }

    override fun observeBin(binId: String): Flow<Bin?> = demoModeStore.mode.flatMapLatest { mode ->
        active(mode).observeBin(binId)
    }

    override fun observeStreamStatus(): Flow<Boolean> = demoModeStore.mode.flatMapLatest { mode ->
        active(mode).observeStreamStatus()
    }

    override fun getEvents(
        binIds: Set<String>,
        localities: Set<String>,
        startTime: Instant,
        endTime: Instant,
    ): Flow<List<WasteEvent>> = demoModeStore.mode.flatMapLatest { mode ->
        active(mode).getEvents(binIds, localities, startTime, endTime)
    }

    override fun streamEvents(): Flow<WasteEvent> = demoModeStore.mode.flatMapLatest { mode ->
        active(mode).streamEvents()
    }

    override suspend fun triggerDemoEvent(binId: String?) {
        active(demoModeStore.mode.value).triggerDemoEvent(binId)
    }

    private fun active(mode: AppMode): BinRepository = when (mode) {
        AppMode.MOCK -> mockRepository
        AppMode.LIVE -> remoteRepository
    }
}
