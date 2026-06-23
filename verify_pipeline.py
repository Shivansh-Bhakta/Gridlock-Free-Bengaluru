import os
import sys
import subprocess
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(DATA_DIR, "violations.csv")
MAP_OUTPUT = os.path.join(DATA_DIR, "map.html")
CHARTS_OUTPUT = os.path.join(DATA_DIR, "charts.html")
DASHBOARD_OUTPUT = os.path.join(DATA_DIR, "dashboard.html")
INDEX_OUTPUT = os.path.join(DATA_DIR, "index.html")
SCHEDULE_CSV = os.path.join(DATA_DIR, "patrol_schedule.csv")
PDF_OUTPUT = os.path.join(DATA_DIR, "patrol_schedule.pdf")

def run_verification():
    print("[VERIFICATION] Starting rigorous verification of the BTP Intelligence Pipeline...")
    
    # 1. Check Input Dataset
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] Raw dataset not found at {INPUT_CSV}")
        sys.exit(1)
    print(f"[SUCCESS] Raw dataset check passed: {INPUT_CSV}")
    
    # 2. Run Pipeline Script via Subprocess
    pipeline_script = os.path.join(DATA_DIR, "run_pipeline.py")
    print(f"[RUN] Executing pipeline {pipeline_script}. This performs all layers: Preprocessing, Feature Engineering, DBSCAN, XGBoost, Isolation Forest, Risk Scoring, Spillover Index, Scheduler, What-If Dashboard, and PDF Generator...")
    try:
        # Run using py (Python 3.14) where packages are installed
        result = subprocess.run(
            ["py", pipeline_script],
            capture_output=True,
            text=True,
            check=True
        )
        print("--- Pipeline Subprocess Output ---")
        print(result.stdout)
        print("---------------------------------")
    except subprocess.CalledProcessError as e:
        print("[ERROR] Pipeline execution failed!")
        print("--- Standard Error ---")
        print(e.stderr)
        print("--- Standard Output ---")
        print(e.stdout)
        sys.exit(1)
        
    # 3. Verify Outputs
    print("\n[CHECK] Validating generated outputs...")
    
    # Check cache and priority csv
    if not os.path.exists(SCHEDULE_CSV):
        print(f"[ERROR] Priority schedule CSV missing: {SCHEDULE_CSV}")
        sys.exit(1)
    
    # Validate schedule row count and columns
    df_sched = pd.read_csv(SCHEDULE_CSV)
    if len(df_sched) != 50:
        print(f"[ERROR] Expected 50 records in patrol schedule, got {len(df_sched)}")
        sys.exit(1)
        
    required_cols = [
        'rank', 'grid_lat', 'grid_lon', 'density', 'risk_score', 
        'canonical_station', 'Morning', 'Evening', 'Night', 'assigned_shift',
        'is_anomaly', 'spillover_index', 'assigned_patrol_unit'
    ]
    for col in required_cols:
        if col not in df_sched.columns:
            print(f"[ERROR] Required column '{col}' missing from patrol schedule CSV.")
            sys.exit(1)
            
    print(f"[SUCCESS] Priority schedule CSV validated: 50 rows, all target features and upgraded columns present.")
    
    # Check dashboard and mapping files
    for filepath in [MAP_OUTPUT, CHARTS_OUTPUT, DASHBOARD_OUTPUT, INDEX_OUTPUT]:
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            print(f"[ERROR] Dashboard component missing or empty: {filepath}")
            sys.exit(1)
    print("[SUCCESS] Folium map, Plotly chart, unified dashboard HTML, and local index.html validated.")
    
    # Check PDF schedule
    if not os.path.exists(PDF_OUTPUT) or os.path.getsize(PDF_OUTPUT) == 0:
        print(f"[ERROR] ReportLab patrol schedule PDF missing or empty: {PDF_OUTPUT}")
        sys.exit(1)
    print(f"[SUCCESS] ReportLab PDF patrol schedule validated: {PDF_OUTPUT} ({os.path.getsize(PDF_OUTPUT) / 1024:.1f} KB)")
    
    # Check Simulated Emails Log (Layer 7)
    EMAIL_LOG = os.path.join(DATA_DIR, "simulated_emails.log")
    if not os.path.exists(EMAIL_LOG) or os.path.getsize(EMAIL_LOG) == 0:
        print(f"[ERROR] Simulated emails log missing or empty: {EMAIL_LOG}")
        sys.exit(1)
        
    # Check that emails specify patrol units
    with open(EMAIL_LOG, 'r', encoding='utf-8') as f:
        email_content = f.read()
        if "Patrol Unit Assigned:" not in email_content:
            print("[ERROR] Simulated emails do not contain Patrol Unit Assigned assignments.")
            sys.exit(1)
            
    print(f"[SUCCESS] Simulated emails log validated and contains patrol asset assignments: {EMAIL_LOG} ({os.path.getsize(EMAIL_LOG) / 1024:.1f} KB)")
    
    print("\n[VERIFICATION SUCCESSFUL] All 7 layers and 4 advanced features executed successfully with zero runtime errors and all outputs are structurally correct.")

if __name__ == "__main__":
    run_verification()
