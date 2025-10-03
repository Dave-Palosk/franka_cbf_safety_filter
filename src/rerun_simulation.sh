#!/bin/bash

# This script runs a specific list of scenarios from a text file
# and forcefully cleans up Gazebo processes between each run.

# --- Configuration ---
PACKAGE_NAME="cbf_safety_filter"
SCENARIO_DIR_NAME="config/generated_scenarios"
RERUN_FILE="src/franka_ros2/cbf_safety_filter/src/rerun_list.txt" # The file containing the list of scenarios to run

# --- Script Logic ---
PACKAGE_DIR=$(ros2 pkg prefix $PACKAGE_NAME)/share/$PACKAGE_NAME
SCENARIO_DIR="$PACKAGE_DIR/$SCENARIO_DIR_NAME"

# Check if the rerun list exists
if [ ! -f "$RERUN_FILE" ]; then
    echo "Error: Rerun list file not found at '$RERUN_FILE'"
    # exit 1
fi

# Read the scenario names into an array
mapfile -t SCENARIOS_TO_RUN < "$RERUN_FILE"

TOTAL_SCENARIOS=${#SCENARIOS_TO_RUN[@]}
CURRENT_SCENARIO=0

ros2 launch cbf_safety_filter rviz.launch.py & RVIZ_PID=$!

# Loop through the specified list of scenarios
for scenario_name in "${SCENARIOS_TO_RUN[@]}"; do
  CURRENT_SCENARIO=$((CURRENT_SCENARIO + 1))
  scenario_file="$SCENARIO_DIR/${scenario_name}.yaml"
  
  # Check if the scenario file actually exists
  if [ ! -f "$scenario_file" ]; then
    echo "--- WARNING: Scenario file not found, skipping: $scenario_file ---"
    continue
  fi

  echo "-----------------------------------------------------------------------"
  echo "--- Running Experiment ($CURRENT_SCENARIO/$TOTAL_SCENARIOS): $scenario_name"
  echo "-----------------------------------------------------------------------"
  
  # Launch the simulation
  timeout 70s ros2 launch cbf_safety_filter gazebo_simulation.launch.py \
    load_gripper:=true \
    franka_hand:='franka_hand' \
    scenario_config_file:="$scenario_file" \
    headless:=True

  pkill -f 'ign gazebo' || true
  sleep 2

  echo "--- Finished $scenario_name. ---"
done

echo "=========================================="
echo "All ${TOTAL_SCENARIOS} specified experiments completed."
echo "=========================================="

# Analyze the results for the new runs
ros2 run cbf_safety_filter analyze_results.py