# SmartBin Android Opus Prompt

Paste the prompt below into Claude Opus or another agentic coding model.

```text
You are a senior staff-level Android and backend product engineer. Your task is to build the full production-quality SmartBin Android application and the supporting backend-facing integration logic around the already-initialized Android Studio project in this repository.

Read this entire prompt carefully and follow the order exactly. Do not jump straight to coding without first understanding the repository context and the existing Android project.

## Phase 0: Mandatory Context Reading Before Any Code Changes

First, read and understand the current repository documentation so you fully understand the goals, product context, technical constraints, and what this project is actually building.

Read these markdown files carefully:
- `README.md`
- `PYTORCH_SETUP.md`
- `REPO_BLUEPRINT.md`
- `SMART_DUSTBIN_PLATFORM_SPEC.md`

Your first responsibility is to internalize:
- what the SmartBin project is
- that the repository started as a waste-classification model repo
- that it is now evolving into a full smart waste-segregating dustbin platform
- that the dustbin performs edge classification and uploads waste events to a backend
- that the Android app is for live map visualization and analytics
- that the final product is a demoable and later deployable smart-bin monitoring platform

After that, inspect the Android project thoroughly.

## Phase 1: Thoroughly Inspect the Existing Android Project

The Android Studio project is already initialized inside:
- `SmartBin_Android/`

Before making any changes, inspect and understand every relevant file and folder inside `SmartBin_Android`.

You must explicitly inspect at least:
- `SmartBin_Android/settings.gradle.kts`
- `SmartBin_Android/build.gradle.kts`
- `SmartBin_Android/gradle/libs.versions.toml`
- `SmartBin_Android/gradle.properties`
- `SmartBin_Android/app/build.gradle.kts`
- `SmartBin_Android/app/src/main/AndroidManifest.xml`
- `SmartBin_Android/app/src/main/java/...`
- `SmartBin_Android/app/src/main/res/...`
- theme files
- current `MainActivity`
- existing tests

You must understand:
- what dependencies already exist
- that this is currently a minimal Compose app scaffold
- what package name is being used
- what needs to be preserved vs replaced

Do not ignore the existing project state. Build on top of it professionally.

## Product Goal

Build a native Android application in Kotlin using Android Studio for the SmartBin system.

This app is for a smart waste-segregating dustbin platform where:
- each physical dustbin has a location pin
- each dustbin runs a waste classifier on the edge, likely Raspberry Pi 4
- each bin uploads events to a backend:
  - waste type
  - timestamp
  - bin identity
  - optional confidence
- the mobile app shows dustbin locations and live analytics

The app must be polished, professional, clean, modern, and convincing in a live demo.
It must not look like a student CRUD app.
Aim for a level of UI taste and polish comparable to clean Google-style internal tools:
- restrained
- precise
- highly legible
- structured
- calm and professional

## Critical Constraints

- Native Android app only
- Kotlin only
- Android Studio project only
- Use Jetpack Compose
- Use Material 3
- No Google Maps
- Use OpenStreetMap-based mapping instead
- Show large obvious dustbin pins on the map
- The app must be free to demo without Google Maps billing/key dependency
- The app must be robust enough for a live demonstration
- Architecture must be production-style, not hacky demo spaghetti

## Primary App Features

### 1. Map Screen

Build a map screen that:
- uses OpenStreetMap
- shows all dustbins as large visible markers/pins
- supports tapping a pin to see details
- supports selecting one bin
- supports selecting multiple bins
- supports selecting bins by locality
- updates live when backend data changes

Each marker should ideally expose:
- bin name
- bin ID
- locality
- online/offline or recent activity indication
- last-seen or last-event timestamp

The map UX must feel intentional and clean.
No ugly default map demo styling.

### 2. Analytics Screen

Build an analytics screen where the user can:
- select one bin
- select multiple bins
- select all bins within a locality
- aggregate selected bins into one combined view

Support time filtering by:
- this week
- this month
- this season
- this year
- custom date range

Season buckets must follow exactly this requested grouping:
- February, March, April
- May, June, July
- August, September, October
- November, December, January

The analytics screen must display:
- total number of waste events
- counts by waste type
- percentages by waste type
- pie chart for waste composition
- bar chart for counts
- optional trend/timeseries if implemented cleanly

If multiple bins are selected:
- aggregate totals first
- compute percentages over the combined totals

### 3. Real-Time Demo Behavior

The app must be able to participate in a live demo where:
- a dustbin bins an item
- a backend receives the event
- the app updates live without manual refresh

So implement:
- realtime update flow
- reconnect-safe behavior
- loading/error states
- graceful fallback if socket stream temporarily disconnects

### 4. Demo-Friendly Backend Contract

You are not implementing the ML model here, but the app and architecture must assume this backend event shape:
- `bin_id`
- `bin_name`
- `locality`
- `latitude`
- `longitude`
- `predicted_class`
- `confidence`
- `event_time`
- `uploaded_at`
- `source_device_id`
- `model_version`

Waste classes for now:
- `metal`
- `organic`
- `other`
- `paper`

## Important Product Understanding

This Android app is not an isolated map app.
It belongs to a larger platform where:
- the ML repo trains the waste classifier
- a Raspberry Pi can run that classifier on the edge
- the physical dustbin uploads waste events
- the app consumes those events and visualizes the fleet

So your implementation choices should reflect:
- IoT fleet dashboard behavior
- analytics clarity
- demo reliability
- future extensibility

## Required Technical Direction

Use the most robust practical stack for the Android app.

Preferred Android stack:
- Kotlin
- Jetpack Compose
- Material 3
- strong layered architecture
- MVVM or Clean Architecture with sane pragmatism
- Hilt for dependency injection
- Retrofit + OkHttp for REST
- WebSocket or SSE for realtime
- Kotlin coroutines
- StateFlow
- Room for local cache if useful
- repository pattern
- clear domain/data/ui separation

For mapping:
- use a stable OpenStreetMap-compatible Android solution
- choose the option most likely to succeed in a real app:
  - osmdroid
  - or MapLibre with OSM tiles
- if one is clearly better for Compose integration and demo reliability, choose it and proceed

For charts:
- use a reliable chart library compatible with Compose
- keep chart visuals polished and not default-ugly

## UI / UX Requirements

The app must feel polished and product-grade.

Design direction:
- clean
- professional
- spacious
- sharp typography
- sensible hierarchy
- good empty/loading/error states
- visually coherent
- not flashy
- not generic “AI slop”

Use:
- bottom navigation or tabbed navigation where appropriate
- grouped features and screens logically
- proper cards, chips, filters, sheets, dialogs, and list states
- a strong map-to-analytics workflow

Avoid:
- cheap placeholder styling
- purple default Compose starter look
- untouched starter theme values
- overcomplicated navigation for no reason

Replace the default starter look completely.

## Architecture Requirements

Design the app as if it will survive beyond a hackathon.

Required characteristics:
- clean package structure
- no God classes
- no business logic inside composables
- reusable UI components
- typed UI state
- isolated network layer
- deterministic filtering logic
- analytics transformations implemented cleanly
- ability to mock backend data for demo mode

Include:
- demo mode
- live mode
- fake repository or simulator source for local preview/testing

## Backend Expectations

If backend-side assumptions or adapter logic are needed, define them professionally.

At minimum the app should be built against endpoints like:
- `GET /bins`
- `GET /bins/{bin_id}`
- `GET /localities`
- `GET /analytics/summary`
- `GET /analytics/composition`
- `GET /analytics/timeseries`
- realtime endpoint via WebSocket or SSE

The app should not depend on a fragile backend shape.
Define DTOs and mapping carefully.

## Required Deliverable Behavior

You are not allowed to stop at planning.

You must actually produce:
- the code changes
- the Android app structure
- all necessary Gradle dependency updates
- architecture setup
- networking layer
- map screen
- analytics screen
- state management
- charting
- realtime flow
- demo/mock mode
- documentation

## Working Rules

- Make the best pragmatic engineering choices and proceed
- If something is ambiguous, choose the simplest professional option
- Preserve what is useful from the existing initialized Android project
- Replace starter boilerplate where appropriate
- Do not leave the app in partial scaffold form
- Build a coherent, runnable Android app

## Concrete Expectations For Output

1. Start by summarizing your understanding of the repo and app goals after reading the markdown files.
2. Summarize what currently exists in `SmartBin_Android`.
3. Propose the final architecture and chosen libraries.
4. Then implement the actual code.
5. Update Gradle files and dependencies as needed.
6. Build out the full app.
7. Add README / setup instructions specific to the Android project.
8. Include notes on how to connect it to a demo backend and how to run in mock mode.

## Specific Functional Details To Support

The app should support:
- fleet overview
- map of all bins
- large dustbin pins
- pin detail bottom sheet or detail card
- multi-select bins
- locality-based filter
- time filters
- this week / this month / this season / this year / custom range
- pie chart and bar chart
- combined analytics across selected bins
- live update behavior
- polished UI

## Quality Bar

Your output must be:
- runnable
- coherent
- maintainable
- presentable in a live demo
- not a toy

If there is a tradeoff between “overly ambitious architecture” and “stable polished demo,” choose the stable polished demo.

Now begin by reading:
- `README.md`
- `PYTORCH_SETUP.md`
- `REPO_BLUEPRINT.md`
- `SMART_DUSTBIN_PLATFORM_SPEC.md`

Then inspect all relevant files inside:
- `SmartBin_Android/`

Only after that should you design and implement the app.
```

