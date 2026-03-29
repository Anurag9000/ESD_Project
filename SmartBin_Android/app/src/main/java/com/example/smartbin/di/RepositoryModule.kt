package com.example.smartbin.di

import com.example.smartbin.data.repository.MockBinRepository
import com.example.smartbin.domain.repository.BinRepository
import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
abstract class RepositoryModule {

    @Binds
    @Singleton
    abstract fun bindBinRepository(
        mockBinRepository: MockBinRepository
    ): BinRepository
}
