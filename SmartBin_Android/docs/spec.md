# SmartBin Android Product Specification

## 1. Goal
The SmartBin Android app is a professional **IoT Fleet Monitoring Dashboard**. It provides real-time geographic visualization and deep waste analytics for a network of smart, segregating dustbins.

## 2. Target Users
- **City Waste Managers:** To monitor bin fill levels and composition across localities.
- **Sustainability Researchers:** To analyze waste trends over time and by material type.
- **Technical Stakeholders:** To verify edge-to-cloud performance during live demonstrations.

## 3. Core Features

### 3.1 Fleet Map (OpenStreetMap)
- **Visuals:** High-contrast markers representing bins.
- **Markers:** 
    - Color-coded or icon-labeled by dominant waste type or status.
    - Real-time "pulse" animation when a new event is binned.
- **Interactions:**
    - **Single Tap:** Open a `DetailCard` or `BottomSheet` showing bin name, ID, locality, and last-seen timestamp.
    - **Multi-Select:** Select multiple bins to aggregate their analytics.
    - **Locality Filtering:** A scrollable row of "Locality Chips" (e.g., "Downtown", "North Park") to quickly filter the map view.

### 3.2 Analytics Dashboard
- **Aggregation Logic:** Metrics (Total Counts, Class Percentages) are aggregated across all selected bins in the specified time range.
- **Time Filters:**
    - Today / This Week / This Month / This Year.
    - **Custom Seasons:**
        - Q1: February, March, April.
        - Q2: May, June, July.
        - Q3: August, September, October.
        - Q4: November, December, January.
    - Custom Date Range picker.
- **Charts:**
    - **Waste Composition:** A professional Donut Chart showing the percentage split of `Metal`, `Organic`, `Paper`, and `Other`.
    - **Waste Trends:** A Grouped Bar Chart showing event counts over time (daily/weekly).

### 3.3 Real-Time Demo Engine
- **Instant Updates:** The app updates its state within 500ms of a backend event notification.
- **Connectivity Status:** A subtle indicator showing if the app is connected to the live stream.
- **Demo/Mock Mode:** A toggle in developer settings (or a specific build variant) to run the app with generated simulated events for presentation stability.

## 4. Waste Categories
The system tracks four primary categories:
1. **Metal**
2. **Organic**
3. **Paper**
4. **Other** (Includes non-metallic recyclables and general waste)

## 5. UI/UX Principles
- **Aesthetic:** "Google-style internal tool" — professional, clean, high-density but legible.
- **Theme:** 
    - **Primary:** Deep Slate / Professional Blue.
    - **Category Colors:** 
        - Metal: Silver/Gray.
        - Organic: Green.
        - Paper: Blue/Light Gray.
        - Other: Yellow/Orange.
- **Accessibility:** High contrast ratios for outdoor monitoring and large touch targets for map interaction.

## 6. Backend Integration Contract
- **Bin Discovery:** `GET /bins`
- **Historical Events:** `GET /events?bin_ids=...&start=...&end=...`
- **Real-time Stream:** `WS /events/stream`
- **Event Payload:**
    - `bin_id` (String)
    - `predicted_class` (Enum: metal, organic, paper, other)
    - `confidence` (Float: 0.0-1.0)
    - `event_time` (ISO 8601 Timestamp)
