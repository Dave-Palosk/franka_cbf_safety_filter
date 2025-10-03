# CBF Safety Filter for Franka FR3 Manipulator

Control Barrier Function (CBF) based safety filter for 7-DOF Franka FR3 robotic manipulator with real-time obstacle avoidance.

## Overview

This package implements a Higher-Order Control Barrier Function (HOCBF) safety filter that ensures collision-free motion for a Franka FR3 robot manipulator. The system can operate in two modes:

- **Integrated Mode**: Complete simulation with all components running together
- **Modular Mode**: Separated components for flexible integration with custom controllers


### Key Features

- Real-time collision avoidance using HOCBF
- Support for static and dynamic obstacles
- Multiple nominal controller options (MPC-based and PD-based)
- Gazebo simulation with RViz visualization
- Automated batch scenario testing
- Comprehensive data logging and analysis


## Table of Contents

- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Modes](#usage-modes)
    - [Integrated Mode](#integrated-mode)
    - [Modular Mode](#modular-mode)
- [Documentation](#documentation)


## System Architecture

The system consists of three main components:

1. **Joint Position Controller**: Hardware interface for commanding the robot
2. **Nominal Controller**: Generates desired motion (MPC or PD-based)
3. **Safety Filter**: Applies HOCBF constraints to ensure collision-free motion

Nominal Controller → Safety Filter → Joint Controller → Robot

## Installation

### Prerequisites

- ROS 2 (Humble or later)
- Python 3.8+
- Gazebo Classic or Gazebo Sim
- Required Python packages:

pip install numpy scipy cvxpy casadi pinocchio pyyaml matplotlib pandas

### Build Instructions

Clone the repository

```bash
cd ~/ros2_ws/src


git clone <repository_url> cbf_safety_filter
```

Install dependencies

```bash
cd ~/ros2_ws


rosdep install --from-paths src --ignore-src -r -y
```

- Build the package

```bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```

- Source the workspace

```bash
source install/setup.bash
```


## Quick Start

### Running a Single Simulation

Run integrated simulation

```bash
ros2 launch cbf_safety_filter gazebo_joint_position_controller.launch.py
load_gripper:=true
franka_hand:='franka_hand'
scenario_config_file:=$(ros2 pkg prefix cbf_safety_filter)/share/cbf_safety_filter/config/scenario_0.yaml
```

To see the obstacles in RViz (Add -> By topic -> select the markers)

The generated plots for analysis should be in install/cbf_safety_filter/lib/cbf_safety_filter/plots/sim

### Running Batch Simulations

Generate scenarios first

```bash
ros2 run cbf_safety_filter generate_scenarios.py
```

Run all scenarios (headless mode for speed)

```bash
cd your_franka_ros2_ws


src/franka_ros2/cbf_safety_filter/src/run_simulation.sh
```

To visualiza in headless mode launch rviz too

```bash
ros2 launch cbf_safety_filter rviz.launch.py
```

Analyze results

```bash
ros2 run cbf_safety_filter analyze_results.py
```

The simulation summary result should be in install/cbf_safety_filter/lib/cbf_safety_filter/plots/sim

## Usage Modes

### Integrated Mode

Use this mode for batch simulations and complete system testing.

**Launch file**: `gazebo_simulation.launch.py`

```bash
ros2 launch cbf_safety_filter gazebo_joint_position_controller.launch.py
load_gripper:=true
franka_hand:='franka_hand'
headless:=True
scenario_config_file:=$(ros2 pkg prefix cbf_safety_filter)/share/cbf_safety_filter/config/scenario_0.yaml
```

**What it launches**:

- Gazebo simulation
- Robot state publisher
- Joint state broadcaster
- Joint position controller
- Safety filter (simulation_HOCBF)
- Nominal controller (integrated in simulation_HOCBF)
- RViz (optional, not in headless mode)


### Modular Mode

Use this mode for custom integration, testing individual components, or integrating with your own controllers.

#### Option 1: Launch All Components Separately

- Terminal 1: Start joint position controller

```bash
ros2 launch cbf_safety_filter gazebo_joint_position_controller.launch.py
```

- Terminal 2: Start Nominal Controller

```bash
ros2 launch cbf_safety_filter mpc_nominal_controller.launch.py
scenario_config_file:=$(ros2 pkg prefix cbf_safety_filter)/share/cbf_safety_filter/config/scenario_0.yaml
```

Alternative: pd_nominal_controller.launch.py

- Terminal 3: Start safety filter

```bash
ros2 launch cbf_safety_filter safety_filter.launch.py
load_gripper:=true
franka_hand:='franka_hand'
scenario_config_file:=$(ros2 pkg prefix cbf_safety_filter)/share/cbf_safety_filter/config/scenario_0.yaml
```


#### Option 2: Use Your Own Nominal Controller

Your controller just needs to publish `Float64MultiArray` messages to `/nominal_joint_acceleration` at 50Hz:

```bash
from std_msgs.msg import Float64MultiArray
```

In your control loop (50 Hz)

```bash
msg = Float64MultiArray()
msg.data = your_computed_accelerations # List of 7 joint accelerations
self.publisher.publish(msg)
```

Then launch only the safety filter:

```bash
ros2 launch cbf_safety_filter safety_filter.launch.py
load_gripper:=true
franka_hand:='franka_hand'
scenario_config_file:=$(ros2 pkg prefix cbf_safety_filter)/share/cbf_safety_filter/config/scenario_0.yaml
```

If you need it to send the command to the manipulator start the joint controller

```bash
ros2 launch cbf_safety_filter gazebo_joint_position_controller.launch.py
```

Or read from the topic of the safety filter `/joint_position_controller/external_commands` and do it with your controller


### Key Parameters

**CBF Parameters:**
- `gamma_js`: Controls convergence rate (higher = faster but more aggressive)
- `beta_js`: Safety margin scaling (higher = more conservative)
- `d_margin`: Additional safety clearance in meters

**Goal Parameters:**
- `goal_ee_pos`: Target end-effector position [x, y, z] in meters
- `goal_tolerance_m`: Distance tolerance for reaching goal (default: 0.02m)
- `goal_settle_time_s`: Time to remain at goal before stopping (default: 2.0s)

## Controllers

### 1. MPC Nominal Controller

**File**: `mpc_nominal_controller.py`

**Characteristics:**
- Planning frequency: 10 Hz (MPC trajectory optimization)
- Tracking frequency: 50 Hz (PD control)

**Launch:**
```
ros2 launch cbf_safety_filter mpc_nominal_controller.launch.py
scenario_config_file:=$(ros2 pkg prefix cbf_safety_filter)/share/cbf_safety_filter/config/scenario_0.yaml
```

### 2. PD Nominal Controller

**File**: `pd_nominal_controller.py`

**Characteristics:**
- Control frequency: 50 Hz (direct operational-space PD)
- Features: Null-space joint limit avoidance

**Launch:**
```
ros2 launch cbf_safety_filter mpc_nominal_controller.launch.py
scenario_config_file:=$(ros2 pkg prefix cbf_safety_filter)/share/cbf_safety_filter/config/scenario_0.yaml
```

### Switching Controllers

Controllers are hot-swappable. Stop one (Ctrl+C) and start another using the same scenario file.

## Project Structure

```

cbf_safety_filter/
├── config/
│   ├── generated_scenarios/         \# Auto-generated scenario files
│   └── scenario_0.yaml              \# Example scenario
├── include/
│   ├── cbf_safety_filter/           \# C++ headers
│   │   ├── default_robot_behavior_utils.hpp
│   │   ├── joint_position_controller.hpp
│   │   └── robot_utils.hpp
│   └── urdf/fr3_robot.urdf          \# Robot URDF (modified for collision points)
├── launch/
│   ├── gazebo_simulation.launch.py  \# Main integrated launch
│   ├── gazebo_joint_position_controller.launch.py  \# Joint controller only
│   ├── mpc_nominal_controller.launch.py            \# MPC controller only
│   ├── pd_nominal_controller.launch.py             \# PD controller only
│   ├── safety_filter.launch.py                     \# Safety filter only
│   └── rviz.launch.py                              \# Visualization only
├── src/
│   ├── safety_filter.py              \# HOCBF implementation
│   ├── mpc_nominal_controller.py     \# MPC + PD tracking controller
│   ├── pd_nominal_controller.py      \# Joint-space PD controller
│   ├── joint_position_controller.cpp \# Hardware interface (C++)
│   ├── generate_scenarios.py         \# Scenario generator
│   ├── analyze_results.py            \# Summary statistics generator
│   ├── run_experiments.sh            \# Batch simulation script
│   └── rerun_experiments.sh          \# Rerun failed scenarios
└── README.md

```

## Key Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/joint_states` | JointState | Robot state feedback (50Hz) |
| `/nominal_joint_acceleration` | Float64MultiArray | Desired accelerations from controller (50Hz) |
| `/joint_position_controller/external_commands` | Float64MultiArray | Commands to joint controller (from safety filter) |
| `/visualization_marker_array` | MarkerArray | Obstacle and trajectory markers for RViz |

## Batch Experiment Workflow

### 1. Generate Scenarios

Edit `generate_scenarios.py` to customize:
- Number of scenarios (default: 1000)
- Workspace bounds
- Obstacle count and size range
- Goal position distribution

Run:
```
ros2 run cbf_safety_filter generate_scenarios.py
```

Scenarios will be created in `install/cbf_safety_filter/share/cbf_safety_filter/config/generated_scenarios/`

### 2. Run Experiments

```
cd your_franka_ros2_ws
src/franka_ros2/cbf_safety_filter/src/run_experiments.sh
```

This automatically:
- Runs all scenarios in headless mode
- Logs results to CSV files
- Saves plots and data

### 3. Rerun Failed Scenarios

List failed scenarios in `rerun_list.txt`:
```
scenario_0042
scenario_0153
scenario_0789
```

Then run:
```
src/franka_ros2/cbf_safety_filter/src/rerun_experiments.sh
```

This launches with RViz for visualization and debugging.

### 4. Analyze Results

```
ros2 run cbf_safety_filter analyze_results.py
```

**Generated outputs:**
- `batch_summary.csv`: Per-scenario summary statistics and overall values (success rates, collision counts...)

**Success criteria:**
- Goal reached within tolerance
- No collisions with obstacles
- Completed within time limit

## Troubleshooting

### Robot Not Moving

Check if accelerations are being published:
```
ros2 topic echo /nominal_joint_acceleration
ros2 topic echo /joint_position_controller/externa_commands
```

Verify goal is reachable and not blocked by obstacles.

### Controller Not Loading

Check controller manager status:
```
ros2 control list_controllers
```

Expected output should show `joint_position_controller` as active.

### Gazebo Crashes

Kill all Gazebo processes and clear cache:
```
pkill -f 'ign gazebo'
```

### No Visualization in RViz

In RViz, manually add markers:
1. Click "Add" button
2. Select "By topic"
3. Choose `/visualization_marker_array`
4. Select `MarkerArray`

Make sure fixed frame: world (top left corner)

## Performance Tips

**For faster batch simulations:**
- Use `headless:=True` in launch file

**For debugging:**
- Use `headless:=False` to see Gazebo GUI
- Launch RViz separately with `rviz.launch.py`
- Monitor results in the generated plots