# SmartBin Android Architecture

## Overview
The SmartBin Android application is built using **Clean Architecture** principles to ensure modularity, testability, and scalability. It follows the **MVVM (Model-View-ViewModel)** pattern for the UI layer and utilizes **Unidirectional Data Flow (UDF)** to manage state.

## Layers

### 1. UI Layer (Jetpack Compose)
- **Composables:** Stateless UI components that describe the view based on state.
- **ViewModels:** Maintain UI state using `StateFlow` and handle user interactions by invoking Domain layer Use Cases.
- **Navigation:** Jetpack Compose Navigation for seamless screen transitions.

### 2. Domain Layer (Pure Kotlin)
- **Use Cases:** Encapsulate specific business logic (e.g., `GetAggregatedAnalyticsUseCase`, `StreamWasteEventsUseCase`).
- **Models:** Domain-specific data structures independent of network or database implementation.
- **Repository Interfaces:** Define the contract for data operations, ensuring the Domain layer remains decoupled from data sources.

### 3. Data Layer
- **Repositories:** Implement the Domain layer's repository interfaces, coordinating between various data sources.
- **Network (Remote):** 
    - **Retrofit:** For REST API communication with the SmartBin backend.
    - **OkHttp (WebSockets):** For real-time waste event streaming.
- **Database (Local):** **Room** for caching bin metadata and locality information to support offline/partial connectivity.
- **Mappers:** Convert Data Transfer Objects (DTOs) from the network/database into Domain models.

## Technology Stack
- **Language:** Kotlin
- **UI Framework:** Jetpack Compose with Material 3
- **Dependency Injection:** Hilt
- **Asynchronous Programming:** Kotlin Coroutines & Flow
- **Mapping:** MapLibre Native for Android (OpenStreetMap tiles)
- **Charts:** Vico
- **Serialization:** Kotlinx Serialization
- **Logging:** Timber

## Real-Time Update Mechanism
The application uses a **WebSocket** connection to listen for new waste events. When an event is received:
1. The `WasteEventStream` emits a new value.
2. The `BinRepository` updates the local state or triggers a fresh fetch.
3. ViewModels observing the repository's Flow update their `StateFlow`.
4. The UI reactively updates (e.g., map marker pulses, charts refresh).

## Testing Strategy
- **Unit Tests:** Focus on Use Case logic, Repository mappings, and ViewModel state transitions using MockK and JUnit 5.
- **UI Tests:** Use Compose Test Rule for verifying UI components and user flows.
- **Demo Mode:** A `MockBinRepository` implementation allows the app to run with simulated data for stable demonstrations without a live backend.
