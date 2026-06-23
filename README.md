# BTP Parking Violation Intelligence & Patrol Dispatch Pipeline

This repository contains the complete implementation of the Bengaluru Traffic Police (BTP) Parking Violation Intelligence and Patrol Dispatch Pipeline. The system is designed to identify high-density violation clusters, forecast next-hour traffic congestion proxies, detect anomalies, apply intersection congestion spillover factors, optimize patrol vehicle allocations, and provide an interactive dispatcher interface.

---

## System Architecture and Processing Layers

The intelligence pipeline is structured into seven distinct layers to process raw data, apply predictive models, optimize resources, and generate user deliverables:

### Layer 1: Preprocessing and Cleaning
* Loads the raw dataset containing parking violation records.
* Drops entirely null columns (such as description, closed_datetime, and action_taken_timestamp).
* Converts timestamps to Indian Standard Time (IST) and derives temporal features (hour, day of week, peak hour indicators).
* Filters and parses JSON violation types to calculate base severity scores.

### Layer 2: Dual Spatial Grid Engineering
* Computes dual spatial grids using coordinate roundings:
  * **High-Resolution Micro-Grid**: Rounded to three decimal places (approximately 110-meter cells) to determine precise target locations for patrol unit dispatches.
  * **Regional Macro-Grid**: Rounded to two decimal places (approximately 1.1-kilometer cells) used to forecast regional hourly trends.

### Layer 3: Machine Learning Models
* **Model A: Spatial Hotspot Clusterer (DBSCAN)**:
  * Filtered to approved records to identify dense spatial violation zones.
  * Uses Haversine distance on radian-converted coordinates.
  * Parameters: epsilon = 0.3 km / 6371 km, min_samples = 10.
  * Caches results locally to make subsequent dashboard load times instantaneous.
* **Model B: Temporal Density Predictor (XGBoost)**:
  * Predicts regional parking violation counts for the next hour to serve as a traffic congestion proxy.
  * Computes 1-hour and 3-hour rolling lag counts using chronological grouping.
  * Evaluated using TimeSeriesSplit (3 folds) to achieve a validation RMSE of approximately 26.08 and MAE of 11.06.
* **Model C: Spatial-Temporal Anomaly Detector (Isolation Forest)**:
  * Runs unsupervised anomaly detection on cell-hour aggregates (density, severity, vehicle weight, SCITA gap count, and peak-hour ratio).
  * Parameters: trained with a 5 percent contamination rate.
  * Identifies anomalous cells that blink distinctly on the Leaflet map and populate a dedicated warning panel.

### Layer 4: Composite Risk Scoring and Congestion Spillover Index
* Calculates a composite base risk score: 40 percent Density + 25 percent Severity + 20 percent Junction proximity + 10 percent Peak Hour ratio + 5 percent Vehicle Weight impact.
* **Congestion Spillover Index**:
  * Utilizes a KDTree neighborhood search to find nearby violations within 300 meters (0.003 degrees).
  * Computes a multiplier: `1.0 + 0.15 * log1p(neighboring_junction_violations)`.
  * Scales the base risk score by this multiplier to prioritize patrols at grids that risk spilling over into major intersections.

### Layer 5: Vehicle-Type Patrol Optimization
* Analyzes local vehicle types inside each grid cell using the statistical mode to deploy optimized patrol assets:
  * **Heavy Tow Truck**: Assigned to grids dominated by heavy vehicles (tankers, buses, maxi-cabs) to clear heavy blockages.
  * **Patrol Jeep**: Assigned to grids dominated by passenger vehicles (cars, autos).
  * **Interceptor Bike**: Assigned to grids dominated by light vehicles (motorcycles, scooters) for high mobility in narrow lanes.

### Layer 6: Dynamic Dashboard and Simulator
* Compiles Folium Leaflet maps and Plotly chart elements into a single-file interactive dashboard.
* **What-If Dispatch Simulator**:
  * Incorporates sliders for Available Patrol Units (5 to 50) and Priority Bias (0.0 to 1.0, Risk vs. Density).
  * recalculates blended scores in JavaScript: `Bias * Risk + (1 - Bias) * Density`.
  * Renders a real-time progress bar computing the Violation Risk Mitigation Score (VRMS) to show the percentage of risk covered by active dispatches.
  * Adjusts map opacity dynamically: dispatched units remain at 0.8 opacity, while standby units fade to 0.15 opacity.

### Layer 7: Synced Notifier System
* Groups schedules by station and shift to generate targeted dispatch briefings.
* Logs simulated emails queued and dispatched 1 hour before shifts start, containing coordinates, expected hourly breakdowns, and assigned vehicle assets.

---

## File Directory Structure

* **run_pipeline.py**: The core script executing all seven processing layers of the intelligence pipeline.
* **verify_pipeline.py**: The automated verification suite checking schema requirements, file existences, and validation asserts.
* **Gridlock-Free Bengaluru.pdf**: The compiled widescreen PDF presentation slides.
* **dashboard.html**: The unified interactive dashboard incorporating map, charts, and the What-If Simulator.
* **map.html**: The standalone Leaflet map with spatial anomaly alerts and overlay filters.
* **charts.html**: The standalone Plotly interactive temporal charts.
* **patrol_schedule.pdf**: The print-ready priority patrol schedule report.
* **patrol_schedule.csv**: The priority schedule dataset containing the top 50 high-risk scheduled hotspots.
* **simulated_emails.log**: Log containing simulated shift briefings and assigned patrol assets.
* **README.md**: Detailed documentation outlining the system architecture, file structures, and setup guidelines.

---

## Setup and Running Instructions

### Prerequisites
Ensure Python is installed on your system. The package dependencies are listed below.

### Installation
Install the required packages using the Python launcher pip module:
```bash
py -m pip install pandas numpy scikit-learn xgboost folium plotly reportlab python-pptx pywin32
```

### Input Dataset
Place the raw dataset `violations.csv` in the same directory as the scripts before running the pipeline.

### Verification Execution
To verify the entire environment and run the pipeline from end-to-end, execute the following command:
```bash
py verify_pipeline.py
```

### Manual Pipeline Execution
To run the full pipeline manually and regenerate all dashboard, map, chart, schedule, and log outputs, execute:
```bash
py run_pipeline.py
```
