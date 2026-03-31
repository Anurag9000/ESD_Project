package com.example.smartbin.di

import com.example.smartbin.data.repository.HybridBinRepository
import com.example.smartbin.domain.repository.BinRepository
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object RepositoryModule {

    @Provides
    @Singleton
    fun provideBinRepository(
        hybridBinRepository: HybridBinRepository,
    ): BinRepository = hybridBinRepository
}
