#!/usr/bin/env bash
set -u

BASE_DIR="/data/repository_code/unified_data"
EWS_HISTORY_DIR="$BASE_DIR/ews_history"
LOG_DIR="$BASE_DIR/logs/ews_history_parallel"
MAX_JOBS=3

mkdir -p "$EWS_HISTORY_DIR" "$LOG_DIR"

STATES=(
UTAH TEXAS
ALABAMA ARIZONA CALIFORNIA DELAWARE GEORGIA IDAHO IOWA LOUISIANA MASSACHUSETTS MISSISSIPPI NEBRASKA NEW_HAMPSHIRE NEW_YORK OHIO PENNSYLVANIA SASKATCHEWAN TENNESSEE VERMONT WEST_VIRGINIA
ALASKA ARKANSAS COLORADO DISTRICT_OF_COLUMBIA GUAM ILLINOIS KANSAS MAINE MICHIGAN MISSOURI NEVADA NEW_JERSEY NORTH_CAROLINA OKLAHOMA PUERTO_RICO SOUTH_CAROLINA VIRGINIA WISCONSIN
ALBERTA BRITISH_COLUMBIA CONNECTICUT FLORIDA HAWAII INDIANA KENTUCKY MARYLAND MINNESOTA MONTANA NEW_BRUNSWICK NEW_MEXICO NORTH_DAKOTA OREGON RHODE_ISLAND SOUTH_DAKOTA WASHINGTON WYOMING
)

run_state() {
  STATE="$1"
  MASK_FILE="$BASE_DIR/mask_inputs/by_state/mask_input_${STATE}.tsv"
  LOG_FILE="$LOG_DIR/${STATE}.log"

  echo "====================================================" | tee "$LOG_FILE"
  echo "RUNNING STATE: $STATE" | tee -a "$LOG_FILE"
  echo "====================================================" | tee -a "$LOG_FILE"

  if [ ! -f "$MASK_FILE" ]; then
    echo "[SKIP] No mask file for $STATE" | tee -a "$LOG_FILE"
    return 0
  fi

  micromamba run -n mrms311 \
    mrms-usgs ews build-history-many \
      --mask-input "$MASK_FILE" \
      --base-dir "$BASE_DIR" \
      --out-dir "$EWS_HISTORY_DIR" \
    >> "$LOG_FILE" 2>&1

  STATUS=$?

  if [ $STATUS -ne 0 ]; then
    echo "[ERROR] Failed state: $STATE" | tee -a "$LOG_FILE"
    return $STATUS
  fi

  echo "[DONE] $STATE" | tee -a "$LOG_FILE"
}

for STATE in "${STATES[@]}"; do
  run_state "$STATE" &

  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    sleep 5
  done
done

wait

echo "===================================================="
echo "ALL STATES FINISHED"
echo "Logs: $LOG_DIR"
echo "===================================================="
SH

chmod +x run_ews_by_state_parallel.sh
