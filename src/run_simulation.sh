#!/bin/bash

# This script runs all scenarios found in the generated_scenarios directory
# and forcefully cleans up Gazebo processes between each run.

# Terminal command
# exec src/franka_ros2/cbf_safety_filter/src/run_simulation.sh

# --- Configuration ---
PACKAGE_NAME="cbf_safety_filter"
SCENARIO_DIR_NAME="config/generated_scenarios"

# --- Script Logic ---
PACKAGE_DIR=$(ros2 pkg prefix $PACKAGE_NAME)/share/$PACKAGE_NAME
SCENARIO_DIR="$PACKAGE_DIR/$SCENARIO_DIR_NAME"

TOTAL_SCENARIOS=$(ls -1 "$SCENARIO_DIR"/*.yaml | wc -l)
CURRENT_SCENARIO=0

# ros2 launch cbf_safety_filter rviz.launch.py & RVIZ_PID=$!

# Loop through all .yaml files in the config directory
for scenario_file in "$SCENARIO_DIR"/*.yaml; do
  CURRENT_SCENARIO=$((CURRENT_SCENARIO + 1))
  scenario_name=$(basename "$scenario_file" .yaml)
  log_file="$LOG_DIR/${scenario_name}.log"
  
  echo "-----------------------------------------------------------------------"
  echo "--- Running Experiment ($CURRENT_SCENARIO/$TOTAL_SCENARIOS): $scenario_name"
  echo "--- Log will be saved to: $log_file"
  echo "-----------------------------------------------------------------------"
  
  # Launch the simulation, redirecting all output to the log file.
  # The 'timeout' command is a safety net in case the simulation hangs.
  timeout 70s ros2 launch cbf_safety_filter gazebo_simulation.launch.py \
    load_gripper:=true \
    franka_hand:='franka_hand' \
    scenario_config_file:="$scenario_file" \
    headless:=True \

  pkill -f 'ign gazebo' || true
  sleep 2

  echo "--- Finished $scenario_name. ---"
done

echo "=========================================="
echo "All $TOTAL_SCENARIOS experiments completed."
echo "=========================================="

pkill rviz

ros2 run cbf_safety_filter analyze_results.py

batch