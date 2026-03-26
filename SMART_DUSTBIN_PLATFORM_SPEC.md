# Smart Waste Dustbin Platform Specification

## 1. Objective

This project is not only a waste-image classification model repository. It is the foundation for a smart waste-segregating dustbin platform.

The intended end product is:

- a physical dustbin with on-device waste classification
- a location-aware fleet of deployed bins
- a shared backend that receives bin events in near real time
- a native Android application that visualizes dustbin locations and waste analytics

The core deployment idea is:

1. the dustbin captures an item
2. the edge device classifies the waste type locally
3. the physical binning action is completed
4. an event is uploaded to a shared server
5. the Android app receives the update and reflects it in analytics and maps

This specification formalizes that full system.

## 2. Product Vision

The platform is intended for a smart waste-segregation use case in which each deployed dustbin behaves as an intelligent IoT node.

Each dustbin should:

- know its own identity
- know its geographic location
- classify waste on-device
- record when a waste item was binned
- upload the waste event to a cloud-connected backend

The Android app should allow an operator, researcher, or stakeholder to:

- see where bins are located
- select a single bin or multiple bins
- filter by time period
- inspect waste composition analytics
- observe live updates during a demonstration or deployment

## 3. Scope

### 3.1 In Scope

- edge waste classification on Raspberry Pi 4
- event upload to a central backend
- Android native app in Kotlin built in Android Studio
- live map view of deployed dustbins
- single-bin and multi-bin analytics
- locality-level aggregation
- time-filtered analytics
- demo-ready real-time synchronization between bin and mobile app

### 3.2 Out of Scope for the First Demonstration

- production cloud hardening
- enterprise auth and IAM
- fleet provisioning at scale
- offline conflict resolution for long network outages
- billing optimization at production scale
- MLOps retraining pipeline automation

The first target is a technically solid live demonstration that proves the full loop.

## 4. High-Level System Architecture

The proposed system has four major layers.

### 4.1 Edge Device Layer

Hardware target:

- Raspberry Pi 4

Responsibilities:

- run the trained waste-classification model locally
- receive image/frame input from the dustbin camera
- infer waste class on-device
- associate the event with:
  - `bin_id`
  - `predicted_waste_type`
  - `timestamp`
  - optionally `confidence`
  - optionally `image_reference`
- send the event to a backend over the network

### 4.2 Backend Layer

A shared server will act as the source of truth for the demo and later deployment.

Responsibilities:

- accept waste-event uploads from edge devices
- store dustbin metadata
- store historical waste-event records
- provide filtered analytics APIs
- broadcast updates to connected clients in real time

### 4.3 Mobile Application Layer

Technology target:

- Android native
- Kotlin
- Android Studio

Responsibilities:

- show deployed dustbins on a map
- allow selection of one or many bins
- query analytics by time period
- display charts and summary metrics
- receive real-time updates from the backend

### 4.4 Map / Geospatial Layer

Responsibilities:

- provide visual geographic context
- allow users to see dustbin placement by locality
- support bin selection and aggregation

## 5. Functional Requirements

## 5.1 Dustbin Map Screen

The app must contain a screen that displays dustbins geographically.

Minimum capabilities:

- show all active dustbins as markers on a map
- show each dustbin’s location live from backend data
- allow tapping a marker to inspect a bin
- allow multi-selection of bins
- allow selection of bins by locality or by manual tap selection

Preferred marker metadata:

- bin name
- bin ID
- locality
- online/offline status
- last event timestamp

## 5.2 Waste Analytics Screen

The app must contain a screen that allows time-filtered analytics for one bin or a group of bins.

The user should be able to:

- select one bin
- select multiple bins
- select all bins in a locality
- aggregate selected bins into a combined view

The analytics view should show:

- total waste events in the selected period
- counts per waste class
- percentages per waste class
- optional trend by day/week/month

Visual outputs should include at minimum:

- pie chart for waste composition percentage
- bar chart for waste counts

## 5.3 Time Filtering

The app must support the following time filters:

- this week
- this month
- this season
- this year
- custom date range

### 5.3.1 Season Logic

The requested seasonal groupings from the product discussion were:

- February, March, April
- May, June, July
- August, September, October
- November, December, January

Important note:

- the conversation labeled the last bucket as `summer`, which is almost certainly a naming mistake
- before implementation, the labels should be confirmed

Recommended handling:

- keep the bucket definitions exactly as discussed
- rename them to neutral labels if needed until the business naming is finalized

For example:

- `Season A`: February-April
- `Season B`: May-July
- `Season C`: August-October
- `Season D`: November-January

## 5.4 Real-Time Updates

During demo and deployment, when a dustbin bins a new item:

- the Raspberry Pi should push the event to the backend immediately
- the backend should persist it
- the backend should notify connected mobile clients
- the Android app should update without requiring a manual refresh

This real-time behavior is central to the demo.

## 5.5 Historical Querying

The backend must support querying historical waste data by:

- bin ID
- multiple bin IDs
- locality
- start timestamp
- end timestamp
- predefined time preset

## 6. Data Model

The backend should at minimum store two logical entity groups:

### 6.1 Dustbin Metadata

Suggested fields:

- `bin_id`
- `bin_name`
- `latitude`
- `longitude`
- `locality`
- `status`
- `last_seen_at`
- `installed_at`
- `firmware_version`

### 6.2 Waste Event

Suggested fields:

- `event_id`
- `bin_id`
- `predicted_class`
- `confidence`
- `event_time`
- `uploaded_at`
- `source_device_id`
- `image_uri` or `image_blob_ref` optional
- `model_version`
- `inference_latency_ms` optional

This structure is enough for demo analytics and also future auditability.

## 7. Recommended Backend Design

For the first version, the simplest strong option is:

- FastAPI backend
- PostgreSQL database
- WebSocket support for real-time events

Why this is a good fit:

- easy to prototype
- easy to run locally or on a small VPS
- clean JSON API support
- easy Android integration
- straightforward real-time channel support

### 7.1 Suggested API Endpoints

- `POST /bins/register`
- `GET /bins`
- `GET /bins/{bin_id}`
- `POST /events`
- `GET /events`
- `GET /analytics/summary`
- `GET /analytics/composition`
- `GET /analytics/timeseries`
- `GET /localities`
- `GET /ws` for real-time subscription

### 7.2 Analytics Query Parameters

Suggested query inputs:

- `bin_ids`
- `locality`
- `from`
- `to`
- `preset`
- `group_by`

Suggested `group_by` options:

- `day`
- `week`
- `month`
- `year`
- `season`

## 8. Android Application Design

## 8.1 Platform

Target:

- native Android
- Kotlin
- Android Studio

Recommended UI stack:

- Jetpack Compose

Reason:

- modern Android-native UI
- easier state-driven screen updates
- easier chart/dashboard composition
- strong long-term maintainability

## 8.2 Recommended App Modules

- `data`
  - API models
  - repositories
  - DTOs
- `network`
  - Retrofit
  - WebSocket client
- `maps`
  - map screen
  - marker state
- `analytics`
  - filters
  - chart preparation
  - summary cards
- `ui`
  - Compose screens
  - navigation
- `domain`
  - use cases for fetching bins and analytics

## 8.3 Recommended Screens

### Screen 1: Live Dustbin Map

Features:

- full-screen map
- dustbin markers
- marker info cards
- single selection
- multi-selection
- locality selection support

### Screen 2: Analytics Dashboard

Features:

- selected bins/locality summary
- date and preset filters
- waste composition pie chart
- waste count bar chart
- total bins selected
- total events in range
- per-class percentages

### Optional Screen 3: Bin Detail

Features:

- single-bin metadata
- recent events
- last online time
- quick jump to analytics

## 9. Real-Time Demonstration Flow

The first full demo should work like this:

1. Raspberry Pi classifies waste locally.
2. Raspberry Pi sends a `waste_event` to the shared backend.
3. Backend stores the event in the database.
4. Backend pushes a real-time notification to mobile clients.
5. Android app receives the update and refreshes:
   - bin last activity
   - current counts
   - charts for selected bins/locality

This is enough to demonstrate:

- on-edge inference
- network integration
- cloud-connected analytics
- live fleet visualization

## 10. Deployment Architecture for Demo

For the demo, the practical objective is not “real cloud at production scale.” The objective is “shared, networked, real-time system behavior.”

A good demo topology is:

- Raspberry Pi 4 running the classifier
- one common backend server
- one Android app connected to the same backend

The shared backend can be:

- a local machine on the same network
- a small VPS
- a cloud VM

This means the system will still behave like a cloud-connected platform even if it is demo-hosted.

## 11. Mapping Stack Decision

The product requirement mentions a Google Maps-based view. This has an important practical constraint.

### 11.1 Google Maps Requirement Status

As of March 26, 2026, official Google Maps Platform documentation states that:

- the Maps SDK for Android requires billing to be enabled on the project
- requests must include an API key or OAuth token
- the SDK uses a pay-as-you-go billing model

Official references:

- https://developers.google.com/maps/documentation/android-sdk/get-api-key
- https://developers.google.com/maps/documentation/android-sdk/usage-and-billing

So the answer to “can we use Google Maps in the Android app with no key and no cost?” is:

- no, not in the standard official Google Maps SDK for Android setup

### 11.2 Recommendation

You have two realistic options:

#### Option A: Google Maps

Use Google Maps if:

- you want the familiar Google map UX
- you are comfortable setting up a Google Cloud project
- you are comfortable using an API key and enabling billing

#### Option B: MapLibre + OpenStreetMap

Use an open stack if:

- you want to avoid Google Maps billing setup for the demo
- you want fewer platform-account dependencies

For a pure demo, this is often the simpler operational choice.

If the requirement is specifically “Google Maps overlay,” then Google Maps should be used, but with the correct expectation that API key setup and billing enablement are part of the implementation.

## 12. Analytics Definitions

For selected bin(s) and selected time period, the app should compute:

- `total_events`
- `count_metal`
- `count_organic`
- `count_other`
- `count_paper`
- `percentage_metal`
- `percentage_organic`
- `percentage_other`
- `percentage_paper`

When multiple bins are selected:

- metrics should be summed first
- percentages should be computed from the combined total

This aligns with the stated requirement that a group selection should show the summed behavior of the selected bins.

## 13. Edge-to-Cloud Event Contract

Suggested payload from Raspberry Pi to backend:

```json
{
  "bin_id": "BIN-001",
  "predicted_class": "paper",
  "confidence": 0.93,
  "event_time": "2026-03-26T18:42:15Z",
  "model_version": "efficientnet_b0_metric_learning_v1",
  "source_device_id": "pi4-001"
}
```

Suggested backend-to-app real-time broadcast:

```json
{
  "type": "waste_event_created",
  "event": {
    "bin_id": "BIN-001",
    "predicted_class": "paper",
    "confidence": 0.93,
    "event_time": "2026-03-26T18:42:15Z"
  }
}
```

## 14. Non-Functional Requirements

### 14.1 Performance

- event upload should feel near-real-time for demo use
- analytics queries for normal date ranges should return quickly
- map should remain responsive with multiple bins visible

### 14.2 Reliability

- edge device should queue or retry if backend is temporarily unavailable
- backend should persist events before broadcasting them
- app should recover cleanly from reconnects

### 14.3 Maintainability

- keep backend API versioned
- keep model version attached to event records
- keep Android app modular

## 15. Suggested Build Order

Recommended implementation sequence:

1. finalize event schema
2. build backend CRUD and analytics endpoints
3. add WebSocket broadcast layer
4. create Android Kotlin app skeleton
5. implement map screen
6. implement analytics dashboard
7. connect Android app to backend
8. connect Raspberry Pi event uploader to backend
9. test live end-to-end demo flow

## 16. Immediate Deliverable Definition

The first successful milestone should demonstrate:

- a Raspberry Pi classifying and uploading a waste event
- a backend storing the event
- an Android app showing dustbin markers on a map
- the app updating analytics in real time for selected bins/localities

That is sufficient to prove the concept of a smart waste-segregating dustbin platform.

## 17. Final Summary

This platform combines:

- edge AI waste classification
- geo-tagged dustbin monitoring
- cloud-connected event storage
- Android-native visualization
- real-time analytics

The repository’s current model-training system is the intelligence core for the bin. The next product layer is the IoT and mobile platform around it.

In short, the intended system is:

- intelligent dustbins in the field
- a common backend server
- a Kotlin Android app for live fleet visualization and analytics
- a demonstration path that works before full production cloudization
