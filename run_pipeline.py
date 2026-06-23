import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KDTree
import xgboost as xgb
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.graph_objects as go
import plotly.offline as pyo

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG & PATHS
# ─────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(DATA_DIR, "violations.csv")
DBSCAN_CACHE = os.path.join(DATA_DIR, "dbscan_labels.npy")
MAP_OUTPUT = os.path.join(DATA_DIR, "map.html")
CHARTS_OUTPUT = os.path.join(DATA_DIR, "charts.html")
DASHBOARD_OUTPUT = os.path.join(DATA_DIR, "dashboard.html")
INDEX_OUTPUT = os.path.join(DATA_DIR, "index.html")

# ─────────────────────────────────────────────
# 1. PREPROCESSING & CLEANING (LAYER 1)
# ─────────────────────────────────────────────
def load_and_preprocess():
    print("[Layer 1] Loading and preprocessing dataset...")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Dataset not found at {INPUT_CSV}")
        
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"    Loaded {len(df):,} records.")
    
    # Drop 3 entirely null columns
    drop_cols = ['description', 'closed_datetime', 'action_taken_timestamp']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    print("    Dropped null columns.")
    
    # Timezone conversion to IST
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], utc=True, errors='coerce').dt.tz_convert('Asia/Kolkata')
    df['hour'] = df['created_datetime'].dt.hour
    df['day_of_week'] = df['created_datetime'].dt.dayofweek
    df['week'] = df['created_datetime'].dt.isocalendar().week
    
    # Safe JSON parsing for violation_type
    def parse_json_array(val):
        if pd.isnull(val):
            return []
        try:
            return json.loads(val)
        except Exception:
            return []
            
    df['violation_list'] = df['violation_type'].apply(parse_json_array)
    print("    Parsed violation lists and set timestamps to IST.")
    return df

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING (LAYER 2)
# ─────────────────────────────────────────────
VEHICLE_WEIGHTS = {
    'TANKER': 5, 'BUS': 4, 'MAXI-CAB': 4,
    'CAR': 3, 'PASSENGER AUTO': 2,
    'MOTOR CYCLE': 1, 'SCOOTER': 1
}

SEVERITY_MAP = {
    "OBSTRUCTING TRAFFIC": 5,
    "PARKING NEAR ROAD CROSSING": 5,
    "DOUBLE PARKING": 4,
    "PARKING IN A MAIN ROAD": 4,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 4,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 4,
    "NO PARKING": 3,
    "WRONG PARKING": 2,
}

def get_shift(hour):
    if 6 <= hour < 14:
        return 'Morning'
    elif 14 <= hour < 22:
        return 'Evening'
    else:
        return 'Night'

def run_feature_engineering(df):
    print("[Layer 2] Engineering features...")
    
    # Vehicle Weight mapping with effective vehicle logic
    df['effective_vehicle'] = df['updated_vehicle_type'].fillna(df['vehicle_type'])
    df['vehicle_weight'] = df['effective_vehicle'].str.upper().str.strip().map(VEHICLE_WEIGHTS).fillna(2.0)
    
    # Dual-resolution spatial grids
    df['grid_lat'] = df['latitude'].round(3)
    df['grid_lon'] = df['longitude'].round(3)
    
    df['xgb_lat'] = df['latitude'].round(2)
    df['xgb_lon'] = df['longitude'].round(2)
    
    # Junction proximity check and weight multiplier (1.5x)
    df['is_junction'] = df['junction_name'].notnull() & (df['junction_name'] != 'No Junction')
    df['junction_multiplier'] = np.where(df['is_junction'], 1.5, 1.0)
    
    # Temporal cycles
    df['peak_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['shift'] = df['hour'].apply(get_shift)
    
    # Severity score calculation
    def calc_severity(v_list):
        if not v_list or not isinstance(v_list, list):
            return 2.0
        return float(sum(SEVERITY_MAP.get(v.upper().strip(), 2.0) for v in v_list))
        
    df['raw_severity_score'] = df['violation_list'].apply(calc_severity)
    df['severity_score'] = df['raw_severity_score'] * df['junction_multiplier']
    
    # Defensible SCITA Gap definition
    df['scita_gap'] = (df['validation_status'] == 'approved') & (df['data_sent_to_scita'].astype(str).str.upper() == 'FALSE')
    
    print("    Features created successfully.")
    return df

# ─────────────────────────────────────────────
# 3. SPATIAL CLUSTERING (LAYER 3A)
# ─────────────────────────────────────────────
def run_dbscan_hotspots(df):
    print("[Layer 3A] Running DBSCAN Hotspot Detection (Approved Records)...")
    
    # Filter to approved validation status
    df_clean = df[df['validation_status'] == 'approved'].copy()
    df_clean = df_clean.dropna(subset=['latitude', 'longitude'])
    
    if len(df_clean) == 0:
        print("    Warning: No approved records found for DBSCAN.")
        df['cluster'] = -1
        return df, pd.DataFrame()
        
    # Check cache for speed
    if os.path.exists(DBSCAN_CACHE):
        print("    Loading cluster labels from cache...")
        labels = np.load(DBSCAN_CACHE)
        if len(labels) == len(df_clean):
            df_clean['cluster'] = labels
        else:
            print("    Cache mismatch. Re-computing DBSCAN...")
            coords = np.radians(df_clean[['latitude', 'longitude']].values)
            db = DBSCAN(eps=0.3/6371, min_samples=10, algorithm='ball_tree', metric='haversine').fit(coords)
            df_clean['cluster'] = db.labels_
            np.save(DBSCAN_CACHE, db.labels_)
    else:
        print("    Computing DBSCAN spatial clustering...")
        coords = np.radians(df_clean[['latitude', 'longitude']].values)
        db = DBSCAN(eps=0.3/6371, min_samples=10, algorithm='ball_tree', metric='haversine').fit(coords)
        df_clean['cluster'] = db.labels_
        np.save(DBSCAN_CACHE, db.labels_)
        
    # Map clusters back to main df (default to -1 for unapproved/noise)
    df['cluster'] = -1
    df.loc[df_clean.index, 'cluster'] = df_clean['cluster']
    
    # Aggregate cluster stats
    hotspots = df_clean[df_clean['cluster'] != -1].groupby('cluster').agg(
        violation_count=('id', 'count'),
        avg_severity=('severity_score', 'mean'),
        peak_hour_pct=('peak_hour', 'mean'),
        scita_gap_count=('scita_gap', 'sum'),
        lat_center=('latitude', 'mean'),
        lon_center=('longitude', 'mean'),
        canonical_station=('police_station', lambda x: x.mode()[0] if not x.empty else 'Unknown')
    ).reset_index()
    
    print(f"    Detected {len(hotspots)} spatial hotspot clusters.")
    return df, hotspots

# ─────────────────────────────────────────────
# 4. TEMPORAL PREDICTION (LAYER 3B)
# ─────────────────────────────────────────────
def run_xgb_temporal_model(df):
    print("[Layer 3B] Training XGBoost Temporal Density Predictor (Congestion Proxy)...")
    
    # Chronological sort
    df_sorted = df.sort_values('created_datetime').reset_index(drop=True)
    
    # Aggregate to 2-decimal grid cells
    grid = df_sorted.groupby(['xgb_lat', 'xgb_lon', 'hour', 'day_of_week']).agg(
        count=('id', 'count'),
        junction_hit=('is_junction', 'max')
    ).reset_index()
    
    # Sort for lag calculations
    grid = grid.sort_values(['xgb_lat', 'xgb_lon', 'day_of_week', 'hour']).reset_index(drop=True)
    
    # Lags using transform to avoid alignment issues
    grid['lag_1h'] = grid.groupby(['xgb_lat', 'xgb_lon'])['count'].transform(lambda x: x.shift(1)).fillna(0.0)
    grid['lag_3h'] = grid.groupby(['xgb_lat', 'xgb_lon'])['count'].transform(lambda x: x.shift(3)).fillna(0.0)
    
    X = grid[['hour', 'day_of_week', 'lag_1h', 'lag_3h', 'junction_hit']].astype(float)
    y = grid['count']
    
    # Time Series Split Validation
    tscv = TimeSeriesSplit(n_splits=3)
    rmses = []
    maes = []
    
    print("    Evaluating XGBRegressor with TimeSeriesSplit...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        m = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42)
        m.fit(X_tr, y_tr)
        preds = np.clip(m.predict(X_val), 0, None)
        
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        rmses.append(rmse)
        maes.append(mae)
        print(f"      Fold {fold+1} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
    print(f"    Average RMSE: {np.mean(rmses):.4f}, MAE: {np.mean(maes):.4f}")
    
    # Final Fit
    model = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)
    model.fit(X, y)
    grid['predicted_density'] = np.clip(model.predict(X), 0, None)
    
    # Map predictions back to original dataframe at 2-decimal level
    pred_map = grid.groupby(['xgb_lat', 'xgb_lon'])['predicted_density'].mean().to_dict()
    df['xgb_pred_density'] = df.set_index(['xgb_lat', 'xgb_lon']).index.map(pred_map).fillna(0.0)
    
    print("    XGBoost model training and forecasting complete.")
    return df

# ─────────────────────────────────────────────
# 4C. ANOMALY DETECTION (LAYER 3C) [ADVANCED FEATURE]
# ─────────────────────────────────────────────
def run_anomaly_detection(df):
    print("[Layer 3C] Training Isolation Forest for Spatial-Temporal Anomaly Detection...")
    
    # Aggregate at 3-decimal cell & hour level
    st_groups = df.groupby(['grid_lat', 'grid_lon', 'hour']).agg(
        density=('id', 'count'),
        avg_severity=('severity_score', 'mean'),
        avg_vehicle_weight=('vehicle_weight', 'mean'),
        scita_gap_count=('scita_gap', 'sum'),
        peak_hour_pct=('peak_hour', 'mean')
    ).reset_index()
    
    # Features for Isolation Forest
    features = st_groups[['density', 'avg_severity', 'avg_vehicle_weight', 'scita_gap_count', 'peak_hour_pct']].fillna(0.0)
    
    # Fit Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    st_groups['anomaly_label'] = iso.fit_predict(features) # -1 is anomaly, 1 is normal
    st_groups['is_anomaly'] = np.where(st_groups['anomaly_label'] == -1, 1, 0)
    
    # Map anomaly label back to cell coordinates (if any hour in cell is anomalous, flag cell)
    cell_anom_map = st_groups.groupby(['grid_lat', 'grid_lon'])['is_anomaly'].max().to_dict()
    
    print(f"    Isolation Forest completed: Flagged {sum(cell_anom_map.values())} anomalous grid cells.")
    return cell_anom_map

# ─────────────────────────────────────────────
# 5. COMPOSITE RISK SCORE & SPILLOVER (LAYER 4) [ADVANCED FEATURE]
# ─────────────────────────────────────────────
def safe_minmax(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - mn) / (mx - mn)

def compute_composite_risk(df, cell_anom_map):
    print("[Layer 4] Computing Composite Risk Score with Congestion Spillover Index...")
    
    # Group at 3-decimal cell level
    grid_cells = df.groupby(['grid_lat', 'grid_lon']).agg(
        density=('id', 'count'),
        severity_sum=('severity_score', 'sum'),
        junction_penalty=('is_junction', 'sum'),
        peak_hour_count=('peak_hour', 'sum'),
        avg_vehicle_weight=('vehicle_weight', 'mean')
    ).reset_index()
    
    # Calculate peak hour weight
    grid_cells['peak_hour_weight'] = grid_cells['peak_hour_count'] / grid_cells['density']
    
    # Formulate Congestion Spillover Index
    # Find neighboring junction violations within 0.003 degrees (~300m) using KDTree
    coords = grid_cells[['grid_lat', 'grid_lon']].values
    tree = KDTree(coords)
    indices = tree.query_radius(coords, r=0.003)
    
    junc_penalties = grid_cells['junction_penalty'].values
    spillover_sums = []
    for idx_list in indices:
        spillover_sums.append(float(junc_penalties[idx_list].sum()))
        
    grid_cells['spillover_penalty'] = spillover_sums
    grid_cells['spillover_index'] = 1.0 + 0.15 * np.log1p(grid_cells['spillover_penalty'])
    
    # Normalize features safely
    density_norm = safe_minmax(grid_cells['density'])
    severity_norm = safe_minmax(grid_cells['severity_sum'])
    junction_norm = safe_minmax(grid_cells['junction_penalty'])
    peak_norm = safe_minmax(grid_cells['peak_hour_weight'])
    vehicle_norm = safe_minmax(grid_cells['avg_vehicle_weight'])
    
    # Risk Score Fusion (Base)
    grid_cells['risk_score_raw'] = (
        density_norm * 0.40 +
        severity_norm * 0.25 +
        junction_norm * 0.20 +
        peak_norm * 0.10 +
        vehicle_norm * 0.05
    )
    
    # Apply Congestion Spillover Multiplier
    grid_cells['risk_score'] = grid_cells['risk_score_raw'] * grid_cells['spillover_index']
    
    # Map Anomaly labels to cells
    grid_cells['is_anomaly'] = grid_cells.set_index(['grid_lat', 'grid_lon']).index.map(cell_anom_map).fillna(0).astype(int)
    
    # Sort and rank
    grid_cells = grid_cells.sort_values('risk_score', ascending=False).reset_index(drop=True)
    grid_cells['rank'] = grid_cells.index + 1
    
    print(f"    Calculated risk scores for {len(grid_cells):,} grid cells.")
    print(f"    Top risk cell score (inflated): {grid_cells['risk_score'].max():.4f}")
    return grid_cells

# ─────────────────────────────────────────────
# 6. PRIORITY SCHEDULER & VEHICLE MATCHING (LAYER 5) [ADVANCED FEATURE]
# ─────────────────────────────────────────────
def build_priority_scheduler(df, grid_cells):
    print("[Layer 5] Generating Enforcement Shift Schedule & Vehicle Asset Optimization...")
    
    # Filter to top 50 cells
    top_50 = grid_cells.head(50).copy()
    
    # Get canonical police station (mode) per 3-decimal cell
    station_map = df.groupby(['grid_lat', 'grid_lon'])['police_station'].agg(
        lambda x: x.mode()[0] if not x.empty else 'Unknown'
    ).reset_index().rename(columns={'police_station': 'canonical_station'})
    
    top_50 = top_50.merge(station_map, on=['grid_lat', 'grid_lon'], how='left')
    
    # Calculate vehicle mode for each cell
    def get_vehicle_mode(g):
        if g.empty:
            return 'CAR'
        return g.mode().iloc[0]
        
    vehicle_modes = df.groupby(['grid_lat', 'grid_lon'])['effective_vehicle'].agg(get_vehicle_mode).reset_index().rename(columns={'effective_vehicle': 'vehicle_mode'})
    top_50 = top_50.merge(vehicle_modes, on=['grid_lat', 'grid_lon'], how='left')
    top_50['vehicle_mode'] = top_50['vehicle_mode'].fillna('CAR')
    
    # Optimize and assign patrol vehicle asset based on vehicle weight categories
    def assign_patrol_unit(mode):
        mode_upper = str(mode).upper().strip()
        if mode_upper in ['TANKER', 'BUS', 'MAXI-CAB', 'TRUCK']:
            return 'Heavy Tow Truck'
        elif mode_upper in ['CAR', 'PASSENGER AUTO', 'THREE WHEELER']:
            return 'Patrol Jeep'
        elif mode_upper in ['MOTOR CYCLE', 'SCOOTER', 'TWO WHEELER']:
            return 'Interceptor Bike'
        else:
            return 'Patrol Jeep'
            
    top_50['assigned_patrol_unit'] = top_50['vehicle_mode'].apply(assign_patrol_unit)
    
    # Calculate per-shift breakdown
    cell_shift = df.groupby(['grid_lat', 'grid_lon', 'shift']).agg(
        shift_count=('id', 'count')
    ).reset_index().pivot_table(
        index=['grid_lat', 'grid_lon'], columns='shift',
        values='shift_count', fill_value=0
    ).reset_index()
    
    # Reindex robustly for shifts
    for s in ['Morning', 'Evening', 'Night']:
        if s not in cell_shift.columns:
            cell_shift[s] = 0
            
    # Merge shift counts to top 50
    top_50 = top_50.merge(cell_shift, on=['grid_lat', 'grid_lon'], how='left').fillna(0)
    
    # Assign target patrol shift based on max shift count
    def assign_shift(row):
        counts = {'Morning': row['Morning'], 'Evening': row['Evening'], 'Night': row['Night']}
        return max(counts, key=counts.get)
        
    top_50['assigned_shift'] = top_50.apply(assign_shift, axis=1)
    
    # Save schedule csv
    schedule_csv = os.path.join(DATA_DIR, "patrol_schedule.csv")
    top_50.to_csv(schedule_csv, index=False)
    print(f"    Patrol schedule saved to {schedule_csv}")
    
    # Print preview
    print("\n[TOP 5] Top 5 Enforcement Targets:")
    for idx, row in top_50.head(5).iterrows():
        print(f"  Rank {row['rank']}: Cell ({row['grid_lat']:.3f}, {row['grid_lon']:.3f}) | Station: {row['canonical_station']} | Risk: {row['risk_score']:.4f} | Anomaly: {row['is_anomaly']} | Patrol: {row['assigned_shift']} ({row['assigned_patrol_unit']})")
        
    return top_50

# ─────────────────────────────────────────────
# 7. DASHBOARD GENERATOR (LAYER 6)
# ─────────────────────────────────────────────
def generate_interactive_map(df, top_50):
    print("[Layer 6] Building Folium Risk Map...")
    
    # Center map on Bengaluru
    bng_center = [12.9716, 77.5946]
    m = folium.Map(location=bng_center, zoom_start=12, tiles='cartodbpositron')
    
    # Add Heatmap Layer
    heat_data = df.dropna(subset=['latitude', 'longitude'])
    heat_list = heat_data[['latitude', 'longitude']].values.tolist()
    HeatMap(heat_list, name='Violation Density Heatmap', min_opacity=0.3, radius=15, blur=10).add_to(m)
    
    # Add Markers for top hotspots
    for idx, row in top_50.iterrows():
        # Color gradient based on rank
        if row['rank'] <= 10:
            color = '#d63031' # Red
            radius = 12
        elif row['rank'] <= 25:
            color = '#e17055' # Orange
            radius = 9
        else:
            color = '#fdcb6e' # Yellow
            radius = 7
            
        # Highlight anomalies
        if row['is_anomaly'] == 1:
            color = '#ff2d55'
            
        anomaly_text = "<b style='color:#ff2d55;'>Anomaly Detected (Isolation Forest)</b><br>" if row['is_anomaly'] == 1 else ""
        
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 220px; font-size:12px;">
            <h4 style="margin: 0 0 5px 0; color:#2d3436;">Rank #{int(row['rank'])} Hotspot</h4>
            {anomaly_text}
            <b>Station:</b> {row['canonical_station']}<br>
            <b>Coordinates:</b> {row['grid_lat']:.3f}, {row['grid_lon']:.3f}<br>
            <b>Risk Score:</b> {row['risk_score']:.4f}<br>
            <b>Spillover Multiplier:</b> {row['spillover_index']:.2f}x<br>
            <b>Violations:</b> {int(row['density'])} total<br>
            <b>Shifts:</b> Morning: {int(row['Morning'])} | Evening: {int(row['Evening'])} | Night: {int(row['Night'])}<br>
            <b style="color:{color};">Assigned Shift: {row['assigned_shift']}</b><br>
            <b>Patrol Unit: {row['assigned_patrol_unit']}</b>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['grid_lat'], row['grid_lon']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"Rank #{int(row['rank'])}: {row['canonical_station']} ({row['assigned_shift']})",
            # Pass custom properties to leaflet options
            grid_lat=float(row['grid_lat']),
            grid_lon=float(row['grid_lon']),
            rank=int(row['rank']),
            is_anomaly=int(row['is_anomaly']),
            assigned_shift=str(row['assigned_shift']),
            assigned_patrol_unit=str(row['assigned_patrol_unit']),
            risk_score=float(row['risk_score']),
            density=int(row['density'])
        ).add_to(m)
        
    # Add beautiful floating legend to the map (no emojis)
    legend_html = '''
    <div style="
        position: fixed; 
        bottom: 30px; left: 30px; width: 190px; height: 125px; 
        z-index: 9999; font-size: 11px; font-family: 'Segoe UI', Arial, sans-serif;
        background-color: #1e272e;
        color: #e3e8f0;
        padding: 10px;
        border-radius: 6px;
        box-shadow: 0 0 15px rgba(0,0,0,0.5);
        border: 1px solid #3d4d5e;
    ">
        <b style="color:#00d2d3; font-size:12px;">Enforcement Priority</b><br>
        <span style="display:inline-block; width:10px; height:10px; background:#d63031; border-radius:50%; margin-right:6px;"></span> Rank 1 - 10 (Critical)<br>
        <span style="display:inline-block; width:10px; height:10px; background:#e17055; border-radius:50%; margin-right:6px;"></span> Rank 11 - 25 (High)<br>
        <span style="display:inline-block; width:10px; height:10px; background:#fdcb6e; border-radius:50%; margin-right:6px;"></span> Rank 26 - 50 (Moderate)<br>
        <span style="display:inline-block; width:10px; height:10px; background:#ff2d55; border-radius:50%; margin-right:6px;"></span> Anomaly Zone (Red Flashing)<br>
        <span style="display:inline-block; width:15px; height:6px; background:#2980b9; margin-right:6px; margin-top:4px; opacity:0.6;"></span> Heatmap: Density Proxy
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
        
    m.save(MAP_OUTPUT)
    print(f"    Map saved to {MAP_OUTPUT}")

def generate_plotly_charts(df, top_50):
    print("[Layer 6] Building Plotly temporal charts...")
    
    # Group by hour for top 5 hotspots
    fig = go.Figure()
    
    for idx, row in top_50.head(5).iterrows():
        cell_data = df[(df['grid_lat'] == row['grid_lat']) & (df['grid_lon'] == row['grid_lon'])]
        hourly_counts = cell_data.groupby('hour')['id'].count().reindex(range(24), fill_value=0)
        
        fig.add_trace(go.Scatter(
            x=list(range(24)),
            y=hourly_counts.values,
            mode='lines+markers',
            name=f"Rank {int(row['rank'])}: {row['canonical_station']} ({row['grid_lat']:.3f}, {row['grid_lon']:.3f})",
            line=dict(width=2.5),
            marker=dict(size=6),
            hovertemplate=(
                f"<b>%{{x}}:00 IST</b><br>"
                f"Violations: %{{y}}<br>"
                f"Station: {row['canonical_station']}<br>"
                f"Coords: {row['grid_lat']:.3f}, {row['grid_lon']:.3f}<extra></extra>"
            )
        ))
        
    fig.update_layout(
        title="Hourly Violation Pattern for Top 5 Hotspot Zones",
        xaxis=dict(title="Hour of Day (IST)", tickmode='linear', tick0=0, dtick=2),
        yaxis=dict(title="Number of Violations"),
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        height=450
    )
    
    pyo.plot(fig, filename=CHARTS_OUTPUT, auto_open=False)
    print(f"    Charts saved to {CHARTS_OUTPUT}")

def build_combined_dashboard(df, top_50, email_data_list=None):
    print("[Layer 6] Merging elements into single file dashboard.html...")
    if email_data_list is None:
        email_data_list = []
    email_json = json.dumps(email_data_list)
    
    # Serialize top 50 cells for the What-If Simulator in JS
    top_50_list = []
    for idx, row in top_50.iterrows():
        top_50_list.append({
            "rank": int(row['rank']),
            "grid_lat": float(row['grid_lat']),
            "grid_lon": float(row['grid_lon']),
            "density": int(row['density']),
            "risk_score": float(row['risk_score']),
            "canonical_station": str(row['canonical_station']),
            "assigned_shift": str(row['assigned_shift']),
            "assigned_patrol_unit": str(row['assigned_patrol_unit']),
            "is_anomaly": int(row['is_anomaly']),
            "Morning": int(row['Morning']),
            "Evening": int(row['Evening']),
            "Night": int(row['Night'])
        })
    top50_json = json.dumps(top_50_list)
    
    # Read Folium map content
    with open(MAP_OUTPUT, 'r', encoding='utf-8') as f:
        map_content = f.read()
        
    # Read Plotly chart content
    with open(CHARTS_OUTPUT, 'r', encoding='utf-8') as f:
        chart_content = f.read()
        
    # Find Leaflet map variable name and static heatmap name
    import re
    match = re.search(r'var\s+(map_[a-f0-9]+)\s*=\s*L\.map', map_content)
    map_var_name = match.group(1) if match else None
    
    match_heat = re.search(r'var\s+(heat_map_[a-f0-9]+)\s*=\s*L\.heatLayer', map_content)
    heat_var_name = match_heat.group(1) if match_heat else None
    
    # Inject script into map_content to publish the map variable and markers list to the parent window
    if map_var_name:
        heat_layer_expose = f"window.parent.staticHeatmapLayer = {heat_var_name};" if heat_var_name else ""
        map_content += f"""
        <script>
            if (window.parent) {{
                window.parent.currentLeafletMap = {map_var_name};
                console.log("Published map instance to parent window.");
                {heat_layer_expose}
                if (window.parent.initMarkers) {{
                    window.parent.initMarkers();
                }}
                if (window.parent.updateLivePatrols) {{
                    window.parent.updateLivePatrols();
                }}
            }}
        </script>
        """
    
    # Base64 encode map and charts to inject cleanly
    import base64
    map_b64 = base64.b64encode(map_content.encode('utf-8')).decode('utf-8')
    chart_b64 = base64.b64encode(chart_content.encode('utf-8')).decode('utf-8')
    
    # Calculate average coordinates (centers) for each police station
    station_centers = df.groupby('police_station')[['latitude', 'longitude']].mean().rename(
        columns={'latitude': 'lat', 'longitude': 'lon'}
    ).to_dict(orient='index')
    station_centers_json = json.dumps(station_centers)

    # Calculate Live Time-Based Risk Data (Layer 6)
    live_df = df.groupby(['grid_lat', 'grid_lon', 'hour', 'day_of_week', 'police_station', 'shift']).size().reset_index(name='count')
    live_df = live_df.sort_values('count', ascending=False)
    live_list = []
    for _, r in live_df.iterrows():
        live_list.append([
            float(r['grid_lat']),
            float(r['grid_lon']),
            int(r['hour']),
            int(r['day_of_week']),
            str(r['police_station']),
            str(r['shift']),
            int(r['count'])
        ])
    live_json = json.dumps(live_list)
    
    # Dynamic table templates will be populated by Javascript on page load
    
    # Validation status counts with clickable stations
    val_counts = df.groupby(['police_station', 'validation_status']).size().unstack(fill_value=0)
    val_counts['Total'] = val_counts.sum(axis=1)
    val_counts = val_counts.sort_values('Total', ascending=False).head(10).reset_index()
    
    val_rows_html = ""
    for idx, row in val_counts.iterrows():
        val_rows_html += f"""
        <tr class="clickable-row" onclick="zoomToStation('{row['police_station']}')">
            <td style="padding: 6px; border-bottom:1px solid #ddd; color:#00d2d3; font-weight:600;">{row['police_station']}</td>
            <td style="padding: 6px; border-bottom:1px solid #ddd; color:green; font-weight:bold;">{row.get('approved', 0)}</td>
            <td style="padding: 6px; border-bottom:1px solid #ddd; color:red;">{row.get('rejected', 0)}</td>
            <td style="padding: 6px; border-bottom:1px solid #ddd; color:orange;">{row.get('pending', 0)}</td>
            <td style="padding: 6px; border-bottom:1px solid #ddd; font-weight:bold;">{row['Total']}</td>
        </tr>
        """
        
    # Defensible SCITA Gap analysis top stations
    scita_gap_df = df[df['scita_gap']].groupby('police_station').size().reset_index(name='gap_count')
    total_approved = df[df['validation_status'] == 'approved'].groupby('police_station').size().reset_index(name='app_count')
    gap_merged = pd.merge(scita_gap_df, total_approved, on='police_station', how='inner')
    gap_merged['gap_percentage'] = (gap_merged['gap_count'] / gap_merged['app_count']) * 100
    gap_merged = gap_merged.sort_values('gap_count', ascending=False).head(10)
    
    gap_rows_html = ""
    for idx, row in gap_merged.iterrows():
        gap_rows_html += f"""
        <tr class="clickable-row" onclick="zoomToStation('{row['police_station']}')">
            <td style="padding: 6px; border-bottom:1px solid #ddd; color:#00d2d3; font-weight:600;">{row['police_station']}</td>
            <td style="padding: 6px; border-bottom:1px solid #ddd; font-weight:bold;">{int(row['gap_count'])}</td>
            <td style="padding: 6px; border-bottom:1px solid #ddd;">{int(row['app_count'])}</td>
            <td style="padding: 6px; border-bottom:1px solid #ddd; color:#d63031; font-weight:bold;">{row['gap_percentage']:.1f}%</td>
        </tr>
        """
        
    # Sidebar rows for Temporal Forecasting Tab (Top 5 hotspots)
    charts_sidebar_rows_html = ""
    for idx, row in top_50.head(5).iterrows():
        s_col = '#3498db' if row['assigned_shift'] == 'Morning' else '#e67e22' if row['assigned_shift'] == 'Evening' else '#9b59b6'
        charts_sidebar_rows_html += f"""
        <tr class="clickable-row" onclick="zoomAndSwitch({row['grid_lat']}, {row['grid_lon']})">
            <td style="padding: 6px; border-bottom:1px solid #ddd; font-weight:bold;">#{int(row['rank'])}</td>
            <td style="padding: 6px; border-bottom:1px solid #ddd; color:#00d2d3; font-weight:600;">{row['canonical_station']}</td>
            <td style="padding: 6px; border-bottom:1px solid #ddd;">{row['grid_lat']:.3f}, {row['grid_lon']:.3f}</td>
            <td style="padding: 6px; border-bottom:1px solid #ddd; font-weight:bold; color:{s_col};">{row['assigned_shift']}</td>
        </tr>
        """

    # Build complete HTML wrapper (fully cleaned of emojis)
    dashboard_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>BTP Parking Enforcement & Congestion Proxy Intelligence</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #0f141c;
            color: #e3e8f0;
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #1e272e, #0f141c);
            padding: 20px;
            border-bottom: 2px solid #3d4d5e;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
            color: #00d2d3;
            letter-spacing: 0.5px;
        }}
        .header p {{
            margin: 5px 0 0 0;
            color: #95a5a6;
            font-size: 14px;
        }}
        .container {{
            display: flex;
            flex-direction: column;
            padding: 15px;
            height: calc(100vh - 110px);
            overflow: hidden;
        }}
        .tabs-header {{
            display: flex;
            margin-bottom: 10px;
            border-bottom: 1px solid #3d4d5e;
        }}
        .tab-btn {{
            cursor: pointer;
            padding: 10px 25px;
            background: none;
            border: none;
            color: #95a5a6;
            font-size: 15px;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        .tab-btn.active {{
            color: #00d2d3;
            border-bottom: 3px solid #00d2d3;
        }}
        .tab-content {{
            display: none;
            flex: 1;
            height: 100%;
            position: relative;
        }}
        .tab-content.active {{
            display: flex;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 2.8fr 1.2fr;
            gap: 15px;
            width: 100%;
            height: calc(100vh - 170px);
            overflow: hidden;
        }}
        .card {{
            background-color: #1e272e;
            border-radius: 8px;
            border: 1px solid #3d4d5e;
            padding: 15px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .card h3 {{
            margin-top: 0;
            color: #00d2d3;
            border-bottom: 1px solid #3d4d5e;
            padding-bottom: 8px;
            font-size: 16px;
        }}
        .map-wrapper, .plotly-wrapper {{
            width: 100%;
            height: 100%;
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid #3d4d5e;
        }}
        .stats-sidebar {{
            display: flex;
            flex-direction: column;
            gap: 15px;
            overflow-y: auto;
            max-height: 100%;
            padding-right: 5px;
        }}
        .stats-sidebar .card {{
            flex-shrink: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            color: #b2bec3;
            text-align: left;
        }}
        th {{
            background-color: #2d3436;
            color: #00d2d3;
            padding: 8px;
            font-weight: 600;
        }}
        .clickable-row {{
            cursor: pointer;
            transition: background-color 0.2s ease;
        }}
        .clickable-row:hover {{
            background-color: #2d3436 !important;
        }}
        .clickable-cell {{
            text-decoration: none;
        }}
        .clickable-row:hover .clickable-cell {{
            text-decoration: underline;
            color: #54a0ff !important;
        }}
        .email-dashboard-grid {{
            display: grid;
            grid-template-columns: 1.2fr 2.8fr;
            gap: 15px;
            width: 100%;
            height: calc(100vh - 170px);
            overflow: hidden;
        }}
        .email-row-active {{
            border-color: #00d2d3 !important;
            background-color: #2c3e50 !important;
        }}
        
        /* What-If Simulator Styling */
        .simulator-slider-container {{
            margin-bottom: 12px;
        }}
        .simulator-slider-label {{
            font-size: 12px;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
        }}
        .slider-input {{
            width: 100%;
            background: #2c3e50;
            border-radius: 4px;
            height: 6px;
            outline: none;
            transition: background 450ms ease-in;
            accent-color: #00d2d3;
        }}
        
        .badge-dispatched {{
            background-color: #2ecc71;
            color: #0f141c;
            padding: 2px 5px;
            border-radius: 3px;
            font-size: 9px;
            font-weight: bold;
        }}
        .badge-standby {{
            background-color: #7f8c8d;
            color: #ffffff;
            padding: 2px 5px;
            border-radius: 3px;
            font-size: 9px;
            font-weight: bold;
            opacity: 0.7;
        }}
        
        .pulse-text-anomaly {{
            color: #ff2d55;
            font-weight: bold;
            animation: text-pulse 1.5s infinite;
        }}
        @keyframes text-pulse {{
            0% {{ opacity: 0.4; }}
            50% {{ opacity: 1.0; }}
            100% {{ opacity: 0.4; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Bengaluru Traffic Police Enforcement Dashboard</h1>
            <p>Composite Parking Violation Risk Map, Anomaly Alerts & What-If Dispatch Simulator</p>
        </div>
        <div style="font-size:12px; text-align:right; color:#95a5a6;">
            <b>IST Timezone Active</b><br>
            Dataset Range: Nov 2023 - Apr 2024
        </div>
    </div>
    
    <div class="container">
        <div class="tabs-header">
            <button class="tab-btn active" onclick="openTab('map-tab', this)">Risk Map (Folium)</button>
            <button class="tab-btn" onclick="openTab('charts-tab', this)">Temporal Forecasting (Plotly)</button>
            <button class="tab-btn" onclick="openTab('emails-tab', this)">Email Dispatch Center</button>
        </div>
        
        <div id="map-tab" class="tab-content active">
            <div class="dashboard-grid">
                <div class="card" style="padding: 0; position: relative;">
                    <!-- Floating Map Overlay Switcher -->
                    <div style="
                        position: absolute;
                        top: 15px;
                        right: 15px;
                        z-index: 1000;
                        background: rgba(30, 39, 46, 0.95);
                        border: 1px solid #3d4d5e;
                        border-radius: 6px;
                        padding: 10px 14px;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
                        backdrop-filter: blur(5px);
                    ">
                        <span style="color:#00d2d3; font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; display: block; margin-bottom: 6px;">Map Overlay Mode</span>
                        <div style="display: flex; gap: 14px; align-items: center; font-size: 12px;">
                            <label style="cursor: pointer; display: flex; align-items: center; gap: 6px; font-weight: 500; color: #e3e8f0;">
                                <input type="radio" name="map-layer" value="live" checked onclick="setMapLayer('live')" style="accent-color: #ff7675; width: 14px; height: 14px; cursor: pointer;">
                                Live Dispatch
                            </label>
                            <label style="cursor: pointer; display: flex; align-items: center; gap: 6px; font-weight: 500; color: #e3e8f0;">
                                <input type="radio" name="map-layer" value="static" onclick="setMapLayer('static')" style="accent-color: #00d2d3; width: 14px; height: 14px; cursor: pointer;">
                                Static All-Time
                            </label>
                        </div>
                    </div>
                    <div class="map-wrapper">
                        <iframe id="map-frame" style="width:100%; height:100%; border:none;"></iframe>
                    </div>
                </div>
                
                <div class="stats-sidebar">
                    <!-- Interactive What-If Dispatch Simulator Card -->
                    <div class="card" style="border: 1px solid #00d2d3; box-shadow: 0 0 10px rgba(0, 210, 211, 0.2);">
                        <h3 style="color:#00d2d3; border-bottom: 1px solid #00d2d3;">What-If Dispatch Simulator</h3>
                        
                        <div class="simulator-slider-container">
                            <div class="simulator-slider-label">
                                <span>Available Patrol Units:</span>
                                <span id="whatif-units-val" style="color:#00d2d3; font-weight:bold;">20</span>
                            </div>
                            <input type="range" id="whatif-units" min="5" max="50" value="20" class="slider-input" oninput="runWhatIfSimulation()">
                        </div>
                        
                        <div class="simulator-slider-container" style="margin-bottom:15px;">
                            <div class="simulator-slider-label">
                                <span>Priority Bias (Risk vs. Density):</span>
                                <span id="whatif-bias-val" style="color:#00d2d3; font-weight:bold;">50% / 50%</span>
                            </div>
                            <input type="range" id="whatif-bias" min="0" max="100" value="50" class="slider-input" oninput="runWhatIfSimulation()">
                        </div>
                        
                        <div style="
                            padding: 10px 12px; 
                            background: #151b24; 
                            border-radius: 6px; 
                            border: 1px solid #3d4d5e;
                            margin-bottom: 10px;
                        ">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 11px; font-weight: 600; color: #95a5a6; text-transform: uppercase;">Violation Risk Mitigation Score:</span>
                                <span id="vrms-val" style="font-size: 16px; font-weight: bold; color: #2ecc71; text-shadow: 0 0 5px rgba(46, 204, 113, 0.4);">0.0%</span>
                            </div>
                            <div style="background-color: #2c3e50; border-radius: 4px; height: 8px; margin-top: 6px; overflow: hidden; border: 1px solid #3d4d5e;">
                                <div id="vrms-bar" style="background: linear-gradient(90deg, #2ecc71, #00d2d3); width: 0%; height: 100%; transition: width 0.2s ease;"></div>
                            </div>
                        </div>
                    </div>
                
                    <!-- Live Dispatch Assistant (Current Time) -->
                    <div class="card" style="border: 1px solid #ff7675;">
                        <h3 style="color:#ff7675; border-bottom: 1px solid #ff7675;">Live Dispatch Assistant</h3>
                        <div style="font-size:12px; margin-bottom:10px; color:#e3e8f0; line-height:1.4;">
                            <span id="live-time-display" style="font-weight:bold; color:#ff7675;">Loading current status...</span><br>
                            <span id="live-count-display">Analyzing active patrol targets...</span>
                        </div>
                        <table id="live-hotspots-table" style="margin-bottom: 8px;">
                            <thead>
                                <tr>
                                    <th>Zone Station</th>
                                    <th>Recent Violations</th>
                                    <th>Active Shift</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr><td colspan="3" style="text-align:center; color:#95a5a6; padding: 15px;">Calculating live targets...</td></tr>
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- Spatial-Temporal Anomaly Alerts Card -->
                    <div class="card" style="border: 1px solid #ff2d55;">
                        <h3 style="color:#ff2d55; border-bottom: 1px solid #ff2d55;">Spatial-Temporal Anomaly Alerts</h3>
                        <p style="font-size:11px; margin-top:-5px; color:#bdc3c7;">
                            Flagged by Isolation Forest (Contamination: 5%)
                        </p>
                        <div id="anomaly-alerts-container" style="max-height: 150px; overflow-y: auto; font-size: 11px;">
                            <!-- Populated dynamically with pulsing alerts -->
                        </div>
                    </div>
                    
                    <div class="card" id="top-10-card">
                        <h3>Top 10 High-Risk Zones</h3>
                        <p style="font-size: 11px; margin-top:-5px; color:#bdc3c7; margin-bottom:5px;">
                            Ranks adjust dynamically to simulator parameters. Click to zoom.
                        </p>
                        
                        <!-- Color Codes Legend -->
                        <div style="font-size:10px; margin-bottom:10px; display:flex; flex-direction:column; gap:4px; border-bottom:1px solid #3d4d5e; padding-bottom:8px; color:#bdc3c7; line-height:1.4;">
                            <div><b>Priority Levels:</b> 
                                <span style="color:#d63031; font-weight:bold; margin-right:6px;">Rank 1-10 Critical</span>
                                <span style="color:#e17055; font-weight:bold; margin-right:6px;">Rank 11-25 High</span>
                                <span style="color:#fdcb6e; font-weight:bold;">Rank 26-50 Moderate</span>
                            </div>
                            <div><b>Patrol Shifts:</b> 
                                <span style="color:#3498db; font-weight:bold; margin-right:6px;">Morning</span>
                                <span style="color:#e67e22; font-weight:bold; margin-right:6px;">Evening</span>
                                <span style="color:#9b59b6; font-weight:bold;">Night</span>
                            </div>
                        </div>
                        <table id="top-10-table">
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Station</th>
                                    <th>Blended Score</th>
                                    <th>Shift</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                <!-- Populated dynamically by JS -->
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="card">
                        <h3>Validation Quality Stats</h3>
                        <p style="font-size: 11px; margin-top:-5px; color:#bdc3c7; margin-bottom:5px;">
                            Click a station to center and zoom map to its jurisdiction
                        </p>
                        <table>
                            <thead>
                                <tr>
                                    <th>Station</th>
                                    <th>Approved</th>
                                    <th>Rejected</th>
                                    <th>Pending</th>
                                    <th>Total</th>
                                </tr>
                            </thead>
                            <tbody>
                                {val_rows_html}
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="card">
                        <h3>Defensible SCITA Enforcement Gap</h3>
                        <p style="font-size: 11px; margin-top:-5px; color:#bdc3c7; margin-bottom:5px;">
                            Click a station to center and zoom map to its leakages
                        </p>
                        <table>
                            <thead>
                                <tr>
                                    <th>Station</th>
                                    <th>Gap Count</th>
                                    <th>Approved</th>
                                    <th>Gap %</th>
                                </tr>
                            </thead>
                            <tbody>
                                {gap_rows_html}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        <div id="charts-tab" class="tab-content">
            <div class="dashboard-grid">
                <div class="card" style="padding: 0;">
                    <div class="plotly-wrapper">
                        <iframe id="charts-frame" style="width:100%; height:100%; border:none;"></iframe>
                    </div>
                </div>
                
                <div class="stats-sidebar">
                    <div class="card">
                        <h3>Temporal Hotspots</h3>
                        <p style="font-size: 11px; margin-top:-5px; color:#bdc3c7; margin-bottom:5px;">
                            Click a row to switch to Map and zoom to that hotspot
                        </p>
                        
                        <!-- Color Codes Legend -->
                        <div style="font-size:10px; margin-bottom:10px; display:flex; flex-direction:column; gap:4px; border-bottom:1px solid #3d4d5e; padding-bottom:8px; color:#bdc3c7; line-height:1.4;">
                            <div><b>Priority Levels:</b> 
                                <span style="color:#d63031; font-weight:bold; margin-right:6px;">Rank 1-10 Critical</span>
                                <span style="color:#e17055; font-weight:bold; margin-right:6px;">Rank 11-25 High</span>
                                <span style="color:#fdcb6e; font-weight:bold;">Rank 26-50 Moderate</span>
                            </div>
                            <div><b>Patrol Shifts:</b> 
                                <span style="color:#3498db; font-weight:bold; margin-right:6px;">Morning</span>
                                <span style="color:#e67e22; font-weight:bold; margin-right:6px;">Evening</span>
                                <span style="color:#9b59b6; font-weight:bold;">Night</span>
                            </div>
                        </div>
                        <table>
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Station</th>
                                    <th>Coordinates</th>
                                    <th>Shift</th>
                                </tr>
                            </thead>
                            <tbody>
                                {charts_sidebar_rows_html}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Automated Email Dispatcher Tab (Fully Cleaned of Emojis) -->
        <div id="emails-tab" class="tab-content">
            <div class="email-dashboard-grid">
                <!-- Left panel: Outbox List -->
                <div class="card" style="padding: 12px; overflow-y: auto; max-height: calc(100vh - 180px);">
                    <h3 style="color:#00d2d3; margin-bottom:10px; border-bottom: 1px solid #3d4d5e; padding-bottom: 8px;">Outbox (Shift Schedule)</h3>
                    <div style="font-size:11px; color:#95a5a6; margin-bottom:12px; line-height: 1.4;">
                        Emails are automatically queued and dispatched 1 hour before shift starts.
                    </div>
                    <div id="email-list-container" style="display: flex; flex-direction: column; gap: 8px;">
                        <!-- Populate dynamically -->
                    </div>
                </div>
                
                <!-- Right panel: Email Reader -->
                <div class="card" style="padding: 15px; overflow-y: auto; max-height: calc(100vh - 180px); background-color: #151b24;">
                    <div id="email-reader-empty" style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color:#95a5a6;">
                        <span style="font-size: 13px; margin-bottom: 10px; border: 1px dashed #3d4d5e; padding: 15px; border-radius: 4px;">[Email Box Empty]</span>
                        Select an email from the outbox to inspect the dispatch message.
                    </div>
                    <div id="email-reader-content" style="display: none;">
                        <div style="border-bottom: 1px solid #3d4d5e; padding-bottom: 12px; margin-bottom: 12px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span id="email-view-status" style="padding: 3px 8px; border-radius: 4px; font-size:10px; font-weight:bold;">ACTIVE</span>
                                <span id="email-view-time" style="font-size: 11px; color:#95a5a6;">Scheduled Trigger: 5:00 AM</span>
                            </div>
                            <h2 id="email-view-subject" style="margin: 0 0 10px 0; color:#e3e8f0; font-size: 16px;">[BTP DISPATCH] Daily Patrol Schedule</h2>
                            <div style="font-size:12px; color:#bdc3c7; line-height: 1.5;">
                                <b>From:</b> dispatch.center@btp.gov.in<br>
                                <b>To:</b> <span id="email-view-to" style="color:#00d2d3;">enforcement.station@btp.gov.in</span>
                            </div>
                        </div>
                        <pre id="email-view-body" style="
                            font-family: 'Courier New', Courier, monospace; 
                            font-size: 12px; 
                            background-color: #0f141c; 
                            color: #a2b4c7; 
                            padding: 12px; 
                            border-radius: 4px; 
                            overflow-x: auto; 
                            white-space: pre-wrap;
                            margin: 0;
                            border: 1px solid #3d4d5e;
                        "></pre>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Setup configurations and variables
        var mapVarName = "{map_var_name}";
        var stationCenters = {station_centers_json};
        var emailData = {email_json};
        var top50Data = {top50_json};
        var leafletMarkers = [];
        
        function getShiftSendTime(shift) {{
            if (shift === 'Morning') return '5:00 AM';
            if (shift === 'Evening') return '1:00 PM';
            return '9:00 PM';
        }}
        
        function updateEmailDashboard() {{
            var now = new Date();
            var currentHour = now.getHours();
            var activeShift = getShiftName(currentHour);
            
            // Assign sorting priority weight to each email based on current shift
            emailData.forEach(function(email) {{
                var shift = email.shift;
                var weight = 2; // Default to ARCHIVED
                if (shift === activeShift) {{
                    weight = 0; // DISPATCHED (ACTIVE)
                }} else {{
                    var isUpcoming = false;
                    if (activeShift === 'Morning' && shift === 'Evening') isUpcoming = true;
                    if (activeShift === 'Evening' && shift === 'Night') isUpcoming = true;
                    if (activeShift === 'Night' && shift === 'Morning') isUpcoming = true;
                    if (isUpcoming) weight = 1; // QUEUED (PENDING)
                }}
                email.sortWeight = weight;
            }});
            
            // Sort emails: active shift first, then upcoming shift, then archived shifts
            emailData.sort(function(a, b) {{
                if (a.sortWeight !== b.sortWeight) {{
                    return a.sortWeight - b.sortWeight;
                }}
                return a.station.localeCompare(b.station);
            }});
            
            var listContainer = document.getElementById("email-list-container");
            if (!listContainer) return;
            listContainer.innerHTML = "";
            
            emailData.forEach(function(email) {{
                var shift = email.shift;
                var station = email.station;
                var sendTime = getShiftSendTime(shift);
                
                var status = "ARCHIVED (SENT)";
                var statusColor = "#95a5a6";
                var weight = email.sortWeight;
                
                if (weight === 0) {{
                    status = "DISPATCHED (ACTIVE)";
                    statusColor = "#2ecc71";
                }} else if (weight === 1) {{
                    status = "QUEUED (PENDING)";
                    statusColor = "#e67e22";
                }}
                
                var emailRow = document.createElement("div");
                emailRow.style.padding = "8px 10px";
                emailRow.style.borderRadius = "4px";
                emailRow.style.background = "#1e272e";
                emailRow.style.border = "1px solid #3d4d5e";
                emailRow.style.cursor = "pointer";
                emailRow.style.transition = "background-color 0.2s";
                emailRow.className = "clickable-row";
                
                emailRow.onclick = function() {{
                    selectEmail(email, status, sendTime, statusColor);
                    document.querySelectorAll(".email-row-active").forEach(el => {{
                        el.style.borderColor = "#3d4d5e";
                        el.classList.remove("email-row-active");
                    }});
                    emailRow.style.borderColor = "#00d2d3";
                    emailRow.classList.add("email-row-active");
                }};
                
                emailRow.innerHTML = `
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-size:10px;">
                        <span style="color:${{statusColor}}; font-weight:bold;">${{status}}</span>
                        <span style="color:#95a5a6;">${{sendTime}}</span>
                    </div>
                    <div style="font-weight:bold; font-size:12px; color:#e3e8f0; margin-bottom:2px;">${{station}} Station</div>
                    <div style="font-size:10px; color:#95a5a6; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                        Shift: ${{shift}} | ${{email.recipient}}
                    </div>
                `;
                listContainer.appendChild(emailRow);
            }});
            
            var activeEmails = emailData.filter(e => e.shift === activeShift);
            if (activeEmails.length > 0 && !document.querySelector(".email-row-active")) {{
                var firstActive = activeEmails[0];
                var sendTime = getShiftSendTime(firstActive.shift);
                selectEmail(firstActive, "DISPATCHED (ACTIVE)", sendTime, "#2ecc71");
                setTimeout(function() {{
                    var rows = listContainer.children;
                    for (var i = 0; i < rows.length; i++) {{
                        if (rows[i].innerText.includes(firstActive.station) && rows[i].innerText.includes(firstActive.shift)) {{
                            rows[i].style.borderColor = "#00d2d3";
                            rows[i].classList.add("email-row-active");
                            break;
                        }}
                    }}
                }}, 100);
            }}
        }}
        
        function selectEmail(email, status, sendTime, statusColor) {{
            document.getElementById("email-reader-empty").style.display = "none";
            document.getElementById("email-reader-content").style.display = "block";
            
            var statusBadge = document.getElementById("email-view-status");
            statusBadge.innerText = status;
            statusBadge.style.backgroundColor = statusColor;
            statusBadge.style.color = "#0f141c";
            
            document.getElementById("email-view-time").innerText = "Scheduled Trigger: " + sendTime;
            document.getElementById("email-view-subject").innerText = email.subject;
            document.getElementById("email-view-to").innerText = email.recipient;
            document.getElementById("email-view-body").innerText = email.body;
        }}
        
        // Base64 decode and write content to iframes
        var mapB64 = "{map_b64}";
        var chartB64 = "{chart_b64}";
        
        try {{
            var mapIframe = document.getElementById('map-frame');
            var mapDoc = mapIframe.contentDocument || mapIframe.contentWindow.document;
            mapDoc.open();
            mapDoc.write(atob(mapB64));
            mapDoc.close();
        }} catch(e) {{
            console.error("Error writing map iframe: ", e);
        }}
        
        try {{
            var chartIframe = document.getElementById('charts-frame');
            var chartDoc = chartIframe.contentDocument || chartIframe.contentWindow.document;
            chartDoc.open();
            chartDoc.write(atob(chartB64));
            chartDoc.close();
        }} catch(e) {{
            console.error("Error writing chart iframe: ", e);
        }}
        
        function zoomTo(lat, lon) {{
            console.log("zoomTo called for coordinates:", lat, lon);
            var map = window.currentLeafletMap;
            if (map) {{
                map.setView([lat, lon], 16, {{ animate: true, duration: 1.5 }});
                console.log("Map view centered successfully.");
                
                var iframe = document.getElementById('map-frame');
                var iframeWindow = iframe.contentWindow || iframe.contentDocument.defaultView;
                if (iframeWindow) {{
                    iframeWindow.setTimeout(function() {{
                        map.eachLayer(function(layer) {{
                            if (layer.getLatLng) {{
                                var latLng = layer.getLatLng();
                                if (Math.abs(latLng.lat - lat) < 0.0001 && Math.abs(latLng.lng - lon) < 0.0001) {{
                                    layer.openPopup();
                                    console.log("Opened popup for marker.");
                                }}
                            }}
                        }});
                    }}, 250);
                }}
            }} else {{
                console.warn("Leaflet map instance not found. Retrying in 200ms...");
                setTimeout(function() {{ zoomTo(lat, lon); }}, 200);
            }}
        }}
        
        function zoomToStation(stationName) {{
            var coords = stationCenters[stationName];
            if (coords) {{
                zoomTo(coords.lat, coords.lon);
            }} else {{
                console.warn("Coordinates not found for station: " + stationName);
            }}
        }}
        
        function zoomAndSwitch(lat, lon) {{
            openTab('map-tab');
            zoomTo(lat, lon);
        }}
        
        function openTab(tabId, btnElement) {{
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});
            document.querySelectorAll('.tab-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            document.getElementById(tabId).classList.add('active');
            if (btnElement) {{
                btnElement.classList.add('active');
            }} else {{
                var btn = document.querySelector("button[onclick*='" + tabId + "']");
                if (btn) btn.classList.add('active');
            }}
        }}

        var liveData = {live_json};
        var isLiveHeatmapOn = true;
        var liveHeatLayerInstance = null;
        var activeMapLayer = 'live';

        function getShiftName(hour) {{
            if (hour >= 6 && hour < 14) {{
                return 'Morning';
            }} else if (hour >= 14 && hour < 22) {{
                return 'Evening';
            }} else {{
                return 'Night';
            }}
        }}

        function setMapLayer(type) {{
            activeMapLayer = type;
            var map = window.currentLeafletMap;
            var staticHeat = window.staticHeatmapLayer;
            
            if (!map) return;
            
            if (type === 'live') {{
                if (staticHeat) {{
                    map.removeLayer(staticHeat);
                }}
                isLiveHeatmapOn = true;
                updateLivePatrols();
            }} else {{
                isLiveHeatmapOn = false;
                if (liveHeatLayerInstance) {{
                    map.removeLayer(liveHeatLayerInstance);
                    liveHeatLayerInstance = null;
                }}
                if (staticHeat) {{
                    map.addLayer(staticHeat);
                }}
            }}
        }}

        function updateLivePatrols() {{
            var now = new Date();
            var currentHour = now.getHours();
            var currentDay = now.getDay();
            
            var pandasDay = currentDay === 0 ? 6 : currentDay - 1;
            var jsDayNames = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
            
            var timeStr = jsDayNames[currentDay] + ", " + formatAMPM(now);
            var timeDisplay = document.getElementById("live-time-display");
            
            var active = liveData.filter(function(item) {{
                return item[2] === currentHour && item[3] === pandasDay;
            }});
            
            var isShiftFallback = false;
            if (active.length < 5) {{
                var shift = getShiftName(currentHour);
                active = liveData.filter(function(item) {{
                    return item[5] === shift && item[3] === pandasDay;
                }});
                isShiftFallback = true;
            }}
            
            active.sort(function(a, b) {{
                return b[6] - a[6];
            }});
            
            if (timeDisplay) {{
                if (isShiftFallback) {{
                    var shift = getShiftName(currentHour);
                    timeDisplay.innerText = "Active Status: " + timeStr + " (Showing " + shift + " Shift due to sparse hourly data)";
                    timeDisplay.style.color = "#ffbe76";
                }} else {{
                    timeDisplay.innerText = "Active Status: " + timeStr;
                    timeDisplay.style.color = "#ff7675";
                }}
            }}
            
            var countDisplay = document.getElementById("live-count-display");
            if (countDisplay) {{
                if (isShiftFallback) {{
                    var shift = getShiftName(currentHour);
                    countDisplay.innerText = active.length + " active cells in this shift (" + shift + ").";
                }} else {{
                    countDisplay.innerText = active.length + " active cells at this hour.";
                }}
            }}
            
            var tbody = document.querySelector("#live-hotspots-table tbody");
            if (tbody) {{
                tbody.innerHTML = "";
                if (active.length === 0) {{
                    tbody.innerHTML = "<tr><td colspan='3' style='text-align:center; color:#95a5a6; padding:15px;'>No active hotspots at this hour.</td></tr>";
                }} else {{
                    var top5 = active.slice(0, 5);
                    top5.forEach(function(item) {{
                        var s_col = item[5] === 'Morning' ? '#3498db' : item[5] === 'Evening' ? '#e67e22' : '#9b59b6';
                        var tr = document.createElement("tr");
                        tr.className = "clickable-row";
                        tr.onclick = function() {{
                            zoomTo(item[0], item[1]);
                        }};
                        tr.innerHTML = `
                            <td style="padding: 6px; border-bottom:1px solid #ddd; color:#ff7675; font-weight:600;">${{item[4]}}</td>
                            <td style="padding: 6px; border-bottom:1px solid #ddd; text-align:center;">${{item[6]}} violations</td>
                            <td style="padding: 6px; border-bottom:1px solid #ddd; font-weight:bold; color:${{s_col}};">${{item[5]}}</td>
                        `;
                        tbody.appendChild(tr);
                    }});
                }}
            }}
            
            drawLiveHeatmap(active);
        }}

        function formatAMPM(date) {{
            var hours = date.getHours();
            var minutes = date.getMinutes();
            var ampm = hours >= 12 ? 'PM' : 'AM';
            hours = hours % 12;
            hours = hours ? hours : 12;
            minutes = minutes < 10 ? '0'+minutes : minutes;
            return hours + ':' + minutes + ' ' + ampm;
        }}

        function drawLiveHeatmap(activeCells) {{
            var map = window.currentLeafletMap;
            if (!map) {{
                setTimeout(function() {{ drawLiveHeatmap(activeCells); }}, 800);
                return;
            }}
            
            var iframe = document.getElementById('map-frame');
            if (!iframe) return;
            var iframeWindow = iframe.contentWindow || iframe.contentDocument.defaultView;
            if (!iframeWindow || !iframeWindow.L) {{
                setTimeout(function() {{ drawLiveHeatmap(activeCells); }}, 500);
                return;
            }}
            var L = iframeWindow.L;
            var staticHeat = window.staticHeatmapLayer;
            
            if (liveHeatLayerInstance) {{
                map.removeLayer(liveHeatLayerInstance);
                liveHeatLayerInstance = null;
            }}
            
            if (activeMapLayer === 'live' && staticHeat) {{
                map.removeLayer(staticHeat);
            }}
            
            if (!isLiveHeatmapOn) {{
                return;
            }}
            
            var heatList = activeCells.map(function(cell) {{
                var intensity = Math.min(cell[6] / 25.0, 1.0);
                return [cell[0], cell[1], intensity];
            }});
            
            if (heatList.length > 0 && L.heatLayer) {{
                liveHeatLayerInstance = L.heatLayer(heatList, {{
                    radius: 28,
                    blur: 18,
                    maxZoom: 15,
                    gradient: {{0.4: '#00d2d3', 0.7: '#a29bfe', 1.0: '#ff7675'}}
                }}).addTo(map);
            }}
        }}
        
        // ────────────────────────────────────────────────────────
        // WHAT-IF SIMULATOR ENGINE (ADVANCED FEATURE)
        // ────────────────────────────────────────────────────────
        function initMarkers() {{
            var map = window.currentLeafletMap;
            if (!map) {{
                setTimeout(initMarkers, 100);
                return;
            }}
            leafletMarkers = [];
            map.eachLayer(function(layer) {{
                if (layer.setStyle && layer.getLatLng) {{
                    leafletMarkers.push(layer);
                }}
            }});
            console.log("Registered Leaflet markers in parent frame: " + leafletMarkers.length);
            
            // Set up anomaly flashing loop
            startAnomalyFlashing();
            
            // Trigger first simulation to initialize dashboard tables
            runWhatIfSimulation();
        }}
        
        function runWhatIfSimulation() {{
            var units = parseInt(document.getElementById("whatif-units").value);
            var biasVal = parseInt(document.getElementById("whatif-bias").value);
            var bias = biasVal / 100.0; // 0.0 means pure density, 1.0 means pure risk
            
            document.getElementById("whatif-units-val").innerText = units;
            document.getElementById("whatif-bias-val").innerText = Math.round(bias * 100) + "% Risk / " + Math.round((1 - bias) * 100) + "% Density";
            
            // Normalized inputs for score blending
            var maxRisk = Math.max(...top50Data.map(d => d.risk_score));
            var minRisk = Math.min(...top50Data.map(d => d.risk_score));
            var maxDensity = Math.max(...top50Data.map(d => d.density));
            var minDensity = Math.min(...top50Data.map(d => d.density));
            
            // Recalculate blended scores
            top50Data.forEach(function(cell) {{
                var normRisk = maxRisk === minRisk ? 0.5 : (cell.risk_score - minRisk) / (maxRisk - minRisk);
                var normDensity = maxDensity === minDensity ? 0.5 : (cell.density - minDensity) / (maxDensity - minDensity);
                
                cell.blendedScore = bias * normRisk + (1 - bias) * normDensity;
            }});
            
            // Sort by blended score descending
            top50Data.sort(function(a, b) {{
                return b.blendedScore - a.blendedScore;
            }});
            
            // Re-assign ranks and dispatch status
            top50Data.forEach(function(cell, idx) {{
                cell.simRank = idx + 1;
                cell.isDispatched = (idx < units);
            }});
            
            // Calculate Violation Risk Mitigation Score (VRMS)
            var sumAll = top50Data.reduce((sum, cell) => sum + cell.blendedScore, 0.0);
            var sumDispatched = top50Data.filter(d => d.isDispatched).reduce((sum, cell) => sum + cell.blendedScore, 0.0);
            var vrms = sumAll === 0 ? 0.0 : (sumDispatched / sumAll) * 100.0;
            
            // Update UI Score Bar
            document.getElementById("vrms-val").innerText = vrms.toFixed(1) + "%";
            document.getElementById("vrms-bar").style.width = vrms.toFixed(1) + "%";
            
            // Update tables and markers
            updateTop10Table();
            updateAnomalyAlerts();
            updateLeafletMapMarkers();
        }}
        
        function updateTop10Table() {{
            var tbody = document.querySelector("#top-10-table tbody");
            if (!tbody) return;
            tbody.innerHTML = "";
            
            var top10 = top50Data.slice(0, 10);
            top10.forEach(function(row) {{
                var s_col = row.assigned_shift === 'Morning' ? '#3498db' : row.assigned_shift === 'Evening' ? '#e67e22' : '#9b59b6';
                var statusBadge = row.isDispatched ? 
                    `<span class="badge-dispatched">DISPATCH</span>` : 
                    `<span class="badge-standby">STANDBY</span>`;
                    
                var tr = document.createElement("tr");
                tr.className = "clickable-row";
                tr.onclick = function() {{ zoomTo(row.grid_lat, row.grid_lon); }};
                
                tr.innerHTML = `
                    <td style="padding: 6px; border-bottom:1px solid #ddd; font-weight:bold;">#${{row.simRank}}</td>
                    <td style="padding: 6px; border-bottom:1px solid #ddd; color:#00d2d3; font-weight:600;">${{row.canonical_station}}</td>
                    <td style="padding: 6px; border-bottom:1px solid #ddd;">${{row.blendedScore.toFixed(3)}}</td>
                    <td style="padding: 6px; border-bottom:1px solid #ddd; font-weight:bold; color:${{s_col}};">${{row.assigned_shift}}</td>
                    <td style="padding: 6px; border-bottom:1px solid #ddd;">${{statusBadge}}</td>
                `;
                tbody.appendChild(tr);
            }});
        }}
        
        function updateAnomalyAlerts() {{
            var container = document.getElementById("anomaly-alerts-container");
            if (!container) return;
            
            var anomalies = top50Data.filter(d => d.is_anomaly === 1);
            if (anomalies.length === 0) {{
                container.innerHTML = `<div style="color:#95a5a6; padding:8px 0; text-align:center;">No anomalies flagged in top zones.</div>`;
                return;
            }}
            
            container.innerHTML = "";
            anomalies.forEach(function(row) {{
                var alertRow = document.createElement("div");
                alertRow.style.padding = "6px 8px";
                alertRow.style.borderBottom = "1px solid #3d4d5e";
                alertRow.style.display = "flex";
                alertRow.style.justify = "space-between";
                alertRow.style.alignItems = "center";
                alertRow.style.cursor = "pointer";
                alertRow.className = "clickable-row";
                alertRow.onclick = function() {{ zoomTo(row.grid_lat, row.grid_lon); }};
                
                alertRow.innerHTML = `
                    <span class="pulse-text-anomaly">ZONE ALERT</span>
                    <span style="color:#e3e8f0; font-weight:600;">${{row.canonical_station}} (${{row.grid_lat.toFixed(3)}}, ${{row.grid_lon.toFixed(3)}})</span>
                    <span style="color:#95a5a6;">Shift: ${{row.assigned_shift}}</span>
                `;
                container.appendChild(alertRow);
            }});
        }}
        
        function updateLeafletMapMarkers() {{
            if (leafletMarkers.length === 0) return;
            
            top50Data.forEach(function(row) {{
                var marker = leafletMarkers.find(function(m) {{
                    var latLng = m.getLatLng();
                    return Math.abs(latLng.lat - row.grid_lat) < 0.0001 &&
                           Math.abs(latLng.lng - row.grid_lon) < 0.0001;
                }});
                
                if (marker) {{
                    var color = '#fdcb6e';
                    var radius = 7;
                    if (row.simRank <= 10) {{
                        color = '#d63031';
                        radius = 12;
                    }} else if (row.simRank <= 25) {{
                        color = '#e17055';
                        radius = 9;
                    }}
                    
                    if (row.is_anomaly === 1) {{
                        color = '#ff2d55';
                    }}
                    
                    var opacity = row.isDispatched ? 0.8 : 0.15;
                    var fillOpacity = row.isDispatched ? 0.7 : 0.08;
                    var weight = row.isDispatched ? 1.5 : 0.5;
                    
                    marker.setStyle({{
                        color: color,
                        fillColor: color,
                        opacity: opacity,
                        fillOpacity: fillOpacity,
                        weight: weight,
                        radius: radius
                    }});
                }}
            }});
        }}
        
        var blinkState = true;
        function startAnomalyFlashing() {{
            setInterval(function() {{
                blinkState = !blinkState;
                leafletMarkers.forEach(function(marker) {{
                    var latLng = marker.getLatLng();
                    var row = top50Data.find(function(d) {{
                        return Math.abs(latLng.lat - d.grid_lat) < 0.0001 &&
                               Math.abs(latLng.lng - d.grid_lon) < 0.0001;
                    }});
                    if (row && row.is_anomaly === 1 && row.isDispatched !== false) {{
                        marker.setStyle({{
                            fillOpacity: blinkState ? 0.9 : 0.25,
                            weight: blinkState ? 3 : 1
                        }});
                    }}
                }});
            }}, 750);
        }}
        
        window.addEventListener('load', function() {{
            setTimeout(function() {{
                updateLivePatrols();
                updateEmailDashboard();
            }}, 1000);
        }});
    </script>
</body>
</html>
"""
    
    with open(DASHBOARD_OUTPUT, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    print(f"[SUCCESS] Integrated dashboard compiled successfully: {DASHBOARD_OUTPUT}")
    
    # Save a copy as index.html
    with open(INDEX_OUTPUT, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    print(f"[SUCCESS] Sync compiled dashboard to: {INDEX_OUTPUT}")

# ─────────────────────────────────────────────
# 7A. EMAIL DISPATCH LAYER (LAYER 7)
# ─────────────────────────────────────────────
def send_patrol_emails(top_50):
    print("[Layer 7] Initializing Email Dispatch Layer (Demo/Simulation)...")
    
    # Map stations to demo emails
    def get_station_email(station):
        clean_name = "".join(c for c in station if c.isalnum()).lower()
        return f"enforcement.{clean_name}@btp.gov.in"
        
    # Group top 50 by canonical station and assigned shift
    grouped = top_50.groupby(['canonical_station', 'assigned_shift'])
    
    simulated_log = []
    email_data_list = []
    
    for (station, shift), group in grouped:
        recipient = get_station_email(station)
        subject = f"[BTP DISPATCH] Daily Patrol Enforcement Schedule - {station} ({shift} Shift)"
        
        # Build email body
        body = f"From: dispatch.center@btp.gov.in\n"
        body += f"To: {recipient}\n"
        body += f"Subject: {subject}\n"
        body += f"Date: 2026-06-23 (IST)\n"
        body += f"========================================================================\n\n"
        body += f"Dear Patrol Officer-in-Charge ({station} Traffic Police Station),\n\n"
        body += f"The BTP Parking Violation Intelligence Pipeline has generated the daily patrol schedule\n"
        body += f"for your jurisdiction for the upcoming {shift} shift. Please prepare enforcers for patrol.\n\n"
        body += f"Patrol Target Details:\n"
        body += f"------------------------------------------------------------------------\n"
        body += f"{'Rank':<6}{'Coordinates':<22}{'Risk Score':<12}{'Expected Violations (M/E/N)':<30}{'Asset Assigned':<20}\n"
        body += f"------------------------------------------------------------------------\n"
        
        for _, row in group.iterrows():
            coords = f"({row['grid_lat']:.3f}, {row['grid_lon']:.3f})"
            violation_str = f"M:{int(row['Morning'])} | E:{int(row['Evening'])} | N:{int(row['Night'])}"
            patrol_unit = str(row['assigned_patrol_unit'])
            body += f"#{int(row['rank']):<5}{coords:<22}{row['risk_score']:<12.4f}{violation_str:<30}{patrol_unit:<20}\n"
            
        body += f"------------------------------------------------------------------------\n\n"
        body += f"Instructions:\n"
        body += f"1. Deploy patrols to the specified coordinate zones during the designated shift hours.\n"
        body += f"2. Focus enforcement on high-weight vehicle violations and intersections within the cells.\n"
        body += f"3. Patrol Unit Assigned: {group.iloc[0]['assigned_patrol_unit']}\n"
        body += f"4. Confirm patrol completion using the SCITA hand-held units.\n\n"
        body += f"Regards,\n"
        body += f"Bengaluru Traffic Police Command & Control Center\n"
        body += f"========================================================================\n"
        
        simulated_log.append(body)
        email_data_list.append({
            "station": station,
            "shift": shift,
            "recipient": recipient,
            "subject": subject,
            "body": body
        })
        
    # Write all simulated emails to simulated_emails.log
    log_path = os.path.join(DATA_DIR, "simulated_emails.log")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(simulated_log))
        
    print(f"    Simulated {len(simulated_log)} emails sent to enforcers.")
    print(f"    All demo emails logged to: {log_path}")
    
    # Print a preview of the first email in the log
    if simulated_log:
        print("\n--- [DEMO EMAIL PREVIEW] ---")
        print("\n".join(simulated_log[0].split("\n")[:15]))
        print("... (truncated preview) ...")
        print("----------------------------\n")
        
    return email_data_list

# ─────────────────────────────────────────────
# 8. MAIN ORCHESTRATOR
# ─────────────────────────────────────────────
def main():
    print("[START] BTP Violation Intelligence Pipeline Running...")
    
    # Layer 1
    df = load_and_preprocess()
    
    # Layer 2
    df = run_feature_engineering(df)
    
    # Layer 3A
    df, hotspots = run_dbscan_hotspots(df)
    
    # Layer 3B
    df = run_xgb_temporal_model(df)
    
    # Layer 3C
    cell_anom_map = run_anomaly_detection(df)
    
    # Layer 4
    grid_cells = compute_composite_risk(df, cell_anom_map)
    
    # Layer 5
    top_50 = build_priority_scheduler(df, grid_cells)
    
    # Layer 6A
    generate_interactive_map(df, top_50)
    generate_plotly_charts(df, top_50)
    
    # PDF Patrol Schedule Generation
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from generate_pdf import build_pdf
        build_pdf()
    except Exception as e:
        print(f"Could not generate PDF schedule: {e}")
            
    # Layer 7: Email Dispatch
    email_data_list = []
    try:
        email_data_list = send_patrol_emails(top_50)
    except Exception as e:
        print(f"Could not run email dispatch: {e}")
        
    # Layer 6B: Build unified dashboard incorporating emails and what-if simulation data
    build_combined_dashboard(df, top_50, email_data_list)
            
    print("\n[FINISHED] Pipeline run finished successfully!")

if __name__ == "__main__":
    main()
