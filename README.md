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
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## System Architecture

The system consists of three main components:

1. **Joint Position Controller**: Hardware interface for commanding the robot
2. **Nominal Controller**: Generates desired motion (MPC or PD-based)
3. **Safety Filter**: Applies HOCBF constraints to ensure collision-free motion

Goal Position → Nominal Controller → Safety Filter → Joint Controller → Robot
↑
Obstacle Detection

text

## Installation

### Prerequisites

- ROS 2 (Humble or later)
- Python 3.8+
- Gazebo Classic or Gazebo Sim
- Required Python packages:

pip install numpy scipy cvxpy casadi pinocchio pyyaml matplotlib pandas

text

### Build Instructions

Clone the repository

cd ~/ros2_ws/src
git clone <repository_url> cbf_safety_filter
Install dependencies

cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
Build the package

colcon build --packages-select cbf_safety_filter
Source the workspace

source install/setup.bash

text

## Quick Start

### Running a Single Simulation

Generate scenarios first

ros2 run cbf_safety_filter generate_scenarios.py
Run integrated simulation

ros2 launch cbf_safety_filter gazebo_simulation.launch.py
scenario_config_file:=install/cbf_safety_filter/share/cbf_safety_filter/config/generated_scenarios/scenario_0000.yaml

text

### Running Batch Simulations

Run all scenarios (headless mode for speed)

cd install/cbf_safety_filter/lib/cbf_safety_filter
./run_simulation.sh
Analyze results

ros2 run cbf_safety_filter analyze_results.py

text

## Usage Modes

### Integrated Mode

Use this mode for batch simulations and complete system testing[web:79].

**Launch file**: `gazebo_simulation.launch.py`

ros2 launch cbf_safety_filter gazebo_simulation.launch.py
scenario_config_file:=path/to/scenario.yaml
headless:=False

text

**What it launches**:
- Gazebo simulation
- Robot state publisher
- Joint state broadcaster
- Joint position controller
- Safety filter (HOCBF)
- Nominal controller (integrated)
- RViz (optional, not in headless mode)

### Modular Mode

Use this mode for custom integration, testing individual components, or integrating with your own controllers[web:76].

#### Option 1: Launch All Components Separately

Terminal 1: Start Gazebo and robot

ros2 launch cbf_safety_filter gazebo_simulation.launch.py
scenario_config_file:=path/to/scenario.yaml
Terminal 2: Start joint position controller

ros2 launch cbf_safety_filter spawn_joint_position_controller.launch.py
Terminal 3: Start safety filter

ros2 launch cbf_safety_filter safety_filter.launch.py
scenario_config_file:=path/to/scenario.yaml
Terminal 4: Start nominal controller (choose one)
Option A: MPC-based controller

ros2 launch cbf_safety_filter mpc_nominal_controller.launch.py
scenario_config_file:=path/to/scenario.yaml
Option B: PD-based controller

ros2 launch cbf_safety_filter pd_nominal_controller.launch.py
scenario_config_file:=path/to/scenario.yaml

text

#### Option 2: Use Your Own Nominal Controller

Your controller just needs to publish `Float64MultiArray` messages to `/nominal_joint_acceleration` at 50Hz:

from std_msgs.msg import Float64MultiArray
In your control loop (50 Hz)

msg = Float64MultiArray()
msg.data = your_computed_accelerations # List of 7 joint accelerations
self.publisher.publish(msg)

text

Then launch only the safety filter:

ros2 launch cbf_safety_filter safety_filter.launch.py
scenario_config_file:=path/to/scenario.yaml

text

## Configuration

### Scenario Files

Scenario files define the goal position, obstacles, and CBF parameters. Located in `config/generated_scenarios/`.

Example structure:

hocbf_controller:
ros__parameters:
goal_ee_pos: [0.3, 0.0, 0.5] # Target end-effector position
gamma_js: 10.0 # CBF gamma parameter
beta_js: 15.0 # CBF beta parameter
d_margin: 0.0 # Safety margin (m)
goal_tolerance_m: 0.02 # Goal reaching tolerance
max_sim_duration_s: 60.0 # Simulation timeout

obstacles:

    name: obstacle_0
    pose_start:
    position: [0.2, 0.3, 0.5]
    size:
    radius: 0.1
    velocity:
    linear: [0.0, 0.0, 0.0]

text

### Updating CBF Parameters

To batch update gamma_js and beta_js across all scenarios:

cd config/generated_scenarios
python update_parameters.py
Follow the prompts

text

## Documentation

Detailed documentation is organized as follows:

- **[USER_GUIDE.md](docs/USER_GUIDE.md)**: Comprehensive usage instructions
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: System design and component details
- **[API.md](docs/API.md)**: Node interfaces, topics, and parameters
- **[CONTROLLERS.md](docs/CONTROLLERS.md)**: Nominal controller implementations
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**: Common issues and solutions

## Project Structure

cbf_safety_filter/
├── config/
│ ├── generated_scenarios/ # Auto-generated scenario files
│ └── cbf_controllers.yaml # Controller configuration
├── include/
│ ├── cbf_safety_filter/ # C++ headers
│ └── urdf/ # Robot URDF files
├── launch/ # Launch files (see below)
├── src/ # Source files
│ ├── safety_filter.py # Main HOCBF implementation
│ ├── mpc_nominal_controller.py
│ ├── pd_nominal_controller.py
│ ├── joint_position_controller.cpp
│ ├── generate_scenarios.py
│ ├── analyze_results.py
│ ├── run_simulation.sh
│ └── rerun_simulation.sh
├── docs/ # Additional documentation
├── CMakeLists.txt
├── package.xml
└── README.md

text

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on Control Barrier Function theory
- Franka Robotics for FR3 robot model
- ROS 2 community

## Citation

If you use this work in your research, please cite:

@software{cbf_safety_filter,
title={CBF Safety Filter for Franka Manipulator},
author={Your Name},
year={2025},
url={repository_url}
}

text
undefined

2. USER_GUIDE.md (docs/USER_GUIDE.md)

text
# User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Running Simulations](#running-simulations)
3. [Modular Usage](#modular-usage)
4. [Visualization](#visualization)
5. [Data Analysis](#data-analysis)

## Getting Started

### First-Time Setup

After building the package, generate initial scenarios:

ros2 run cbf_safety_filter generate_scenarios.py

text

This creates 1000 scenario files in the install directory.

### Understanding Scenarios

Each scenario defines:
- **Goal position**: Target end-effector location
- **Obstacles**: Static or dynamic spherical obstacles
- **CBF parameters**: Safety filter tuning

## Running Simulations

### Single Scenario

**With visualization**:

ros2 launch cbf_safety_filter gazebo_simulation.launch.py
scenario_config_file:=<path_to_scenario>
headless:=False

text

**Headless (faster)**:

ros2 launch cbf_safety_filter gazebo_simulation.launch.py
scenario_config_file:=<path_to_scenario>
headless:=True

text

### Batch Scenarios

The `run_simulation.sh` script automatically runs all scenarios:

cd install/cbf_safety_filter/lib/cbf_safety_filter
./run_simulation.sh

text

Results are logged to `install/cbf_safety_filter/share/cbf_safety_filter/plots/sim/`

### Rerunning Failed Scenarios

If some scenarios fail, list them in `rerun_list.txt`:

scenario_0042
scenario_0153
scenario_0789

text

Then run:

./rerun_simulation.sh

text

This launches with RViz for debugging.

## Modular Usage

### Using MPC Controller

The MPC controller plans trajectories at 10Hz and tracks them with PD at 50Hz:

ros2 launch cbf_safety_filter mpc_nominal_controller.launch.py
scenario_config_file:=path/to/scenario.yaml
mpc_frequency:=10.0
pid_frequency:=50.0

text

**Parameters**:
- `mpc_frequency`: MPC replanning rate (default: 10 Hz)
- `pid_frequency`: PD tracking rate (default: 50 Hz)
- `goal_tolerance_m`: Distance tolerance for goal (default: 0.02 m)
- `goal_settle_time_s`: Time at goal before stopping (default: 2.0 s)

### Using PD Controller

The PD controller directly computes accelerations at 50Hz:

ros2 launch cbf_safety_filter pd_nominal_controller.launch.py
scenario_config_file:=path/to/scenario.yaml
control_frequency:=50.0

text

**Parameters**:
- `control_frequency`: Control loop rate (default: 50 Hz)
- `goal_tolerance_m`: Goal reaching tolerance
- `goal_settle_time_s`: Settling time requirement

### Integration with Custom Controllers

To use your own controller with the safety filter:

1. **Subscribe to** `/joint_states` to get current robot state
2. **Publish to** `/nominal_joint_acceleration` (Float64MultiArray, 50Hz)
   - Array of 7 joint accelerations in rad/s²
3. **Launch safety filter**:

ros2 launch cbf_safety_filter safety_filter.launch.py
scenario_config_file:=path/to/scenario.yaml

text

The safety filter will:
- Subscribe to your `/nominal_joint_acceleration`
- Apply HOCBF constraints
- Publish safe commands to `/safe_joint_acceleration`

## Visualization

### RViz

Launch RViz separately:

ros2 launch cbf_safety_filter rviz.launch.py

text

**Displays**:
- Robot model
- End-effector trajectory
- Obstacles (as spheres)
- Goal marker

### Real-time Monitoring

Monitor topics during simulation:

Joint states

ros2 topic echo /joint_states
Nominal accelerations (from controller)

ros2 topic echo /nominal_joint_acceleration
Safe accelerations (from CBF filter)

ros2 topic echo /safe_joint_acceleration
Goal reached status

ros2 topic echo /goal_reached

text

## Data Analysis

After batch simulations, analyze results:

ros2 run cbf_safety_filter analyze_results.py

text

This generates:
- `summary_statistics.csv`: Success rates, collision statistics
- `scenario_results.csv`: Per-scenario outcomes
- Plots in the `plots/sim/` directory

### Understanding Results

**Success criteria**:
- Goal reached within tolerance
- No collisions with obstacles
- Completed within time limit

**Logged data** (per scenario):
- Joint positions, velocities, accelerations
- End-effector trajectory
- CBF values over time
- Obstacle distances
- Computational timing

## Advanced Topics

### Generating Custom Scenarios

Edit `generate_scenarios.py` to customize:
- Number of scenarios
- Workspace bounds
- Obstacle count and size range
- Goal position distribution

### Tuning CBF Parameters

The key parameters are:
- **gamma_js**: Controls convergence rate (higher = faster but more aggressive)
- **beta_js**: Safety margin scaling (higher = more conservative)
- **d_margin**: Additional safety clearance (meters)

Update all scenarios:

cd config/generated_scenarios
python update_parameters.py

text

### Performance Optimization

For faster batch simulations:
- Use headless mode (`headless:=True`)
- Reduce logging frequency in code
- Run on a powerful machine
- Consider parallel execution (modify scripts)

3. ARCHITECTURE.md (docs/ARCHITECTURE.md)

text
# System Architecture

## Overview

The CBF safety filter system implements a modular control architecture with three main components that communicate via ROS 2 topics[web:75][web:76].

## Component Diagram

┌─────────────────────────────────────────────────────────────┐
│ Gazebo Simulation │
│ (Franka FR3 Robot) │
└───────────────┬─────────────────────────────┬───────────────┘
│ /joint_states │ /joint_commands
│ │
┌───────────▼──────────┐ ┌───────────▼──────────────┐
│ Nominal Controller │ │ Joint Position Controller │
│ - MPC Planner │ │ (Hardware Interface) │
│ - PD Tracker │ └───────────▲──────────────┘
│ or │ │
│ - JS PD Controller │ │ /safe_joint_acceleration
└───────────┬──────────┘ │
│ /nominal_joint_acceleration│
│ │
┌──────▼────────────────────────────┴────┐
│ Safety Filter (HOCBF) │
│ - QP Solver │
│ - Collision Detection │
│ - CBF Constraint Generation │
└────────────────────────────────────────┘

text

## Components

### 1. Joint Position Controller

**File**: `joint_position_controller.cpp`  
**Type**: C++ ROS2 Controller Plugin  
**Frequency**: 50 Hz (commanded)

**Responsibilities**:
- Receives safe joint accelerations from safety filter
- Integrates accelerations to positions via exponential filter
- Sends position commands to Gazebo/hardware interface

**Topics**:
- Subscribes: `/safe_joint_acceleration` (Float64MultiArray)
- Publishes: Joint commands to `controller_manager`

### 2. Nominal Controllers

Two options available:

#### A. MPC Nominal Controller

**File**: `mpc_nominal_controller.py`  
**Frequencies**: 
- MPC planning: 10 Hz
- PD tracking: 50 Hz

**Responsibilities**:
- Plans collision-free trajectory using Model Predictive Control
- Tracks planned trajectory with PD controller
- Publishes desired joint accelerations

**Algorithm**:
1. Reads goal from scenario file
2. Computes MPC trajectory (7-step horizon)
3. PD tracks current step of trajectory
4. Publishes nominal acceleration

**Topics**:
- Subscribes: `/joint_states` (JointState)
- Publishes: `/nominal_joint_acceleration` (Float64MultiArray)
- Publishes: `/goal_reached` (Bool)

#### B. Joint-Space PD Controller

**File**: `pd_nominal_controller.py`  
**Frequency**: 50 Hz

**Responsibilities**:
- Direct operational-space PD control to Cartesian goal
- Null-space joint limit avoidance
- Publishes desired joint accelerations

**Algorithm**:
1. Compute end-effector error (current vs goal)
2. Calculate Jacobian pseudo-inverse
3. Primary task: Cartesian space control
4. Secondary task: Joint limit repulsion (null-space)
5. Combine tasks and publish

**Topics**:
- Subscribes: `/joint_states` (JointState)
- Publishes: `/nominal_joint_acceleration` (Float64MultiArray)
- Publishes: `/goal_reached` (Bool)

### 3. Safety Filter (HOCBF)

**File**: `safety_filter.py`  
**Frequency**: 50 Hz

**Responsibilities**:
- Monitors obstacles and robot state
- Computes CBF constraints
- Solves QP to find safe acceleration
- Ensures collision-free motion

**Algorithm**:
1. Receive nominal acceleration from controller
2. Compute obstacle distances for all collision points
3. Generate CBF constraints (h(x) ≥ 0)
4. Formulate QP: minimize ||ddq_safe - ddq_nominal||²
   - Subject to: CBF constraints
   - Subject to: Joint limits
5. Solve QP and publish safe acceleration

**CBF Formulation**:

h(x) = distance_to_obstacle - d_margin
ḣ(x) ≥ -α₁(h(x))
ḧ(x) ≥ -α₂(ḣ(x)) - α₁(ḣ(x))

where: α₁(s) = gamma * s
α₂(s) = beta * s

text

**Topics**:
- Subscribes: `/joint_states` (JointState)
- Subscribes: `/nominal_joint_acceleration` (Float64MultiArray)
- Publishes: `/safe_joint_acceleration` (Float64MultiArray)
- Publishes: Markers for visualization

## Data Flow

    Robot State Update (50 Hz)
    Gazebo → /joint_states → All Nodes

    Nominal Control (50 Hz)
    Controller → /nominal_joint_acceleration → Safety Filter

    Safety Filtering (50 Hz)
    Safety Filter → /safe_joint_acceleration → Joint Controller

    Command Execution (50 Hz)
    Joint Controller → Gazebo

    Visualization (10 Hz)
    Safety Filter → RViz markers

text

## Collision Detection

**Collision Points**: 
- 7 link centers
- 7 link endpoints  
- End-effector
- Total: 15 points monitored per obstacle

**Distance Computation**:
- Uses Pinocchio forward kinematics
- Computes Euclidean distance to obstacle centers
- Accounts for obstacle radius

## QP Solver

**Library**: CVXPY with OSQP backend

**QP Formulation**:

minimize: ||ddq_safe - ddq_nominal||²

subject to: ḧ(x) + α₂(ḣ) + α₁(h) ≥ 0 (for each obstacle)
ddq_min ≤ ddq_safe ≤ ddq_max
Joint velocity limits (soft)
Joint position limits (soft)

text

## Launch File Architecture

### Integrated Mode
`gazebo_simulation.launch.py` launches everything in sequence:
1. Gazebo + robot spawn
2. Joint state broadcaster
3. Joint position controller  
4. Safety filter
5. Nominal controller (embedded)

### Modular Mode
Separate launch files for each component:
- `spawn_joint_position_controller.launch.py`
- `mpc_nominal_controller.launch.py`
- `pd_nominal_controller.launch.py`
- `safety_filter.launch.py`
- `rviz.launch.py`

## Thread Safety

All nodes use ROS 2's executor model:
- Single-threaded execution within each node
- No shared memory between nodes
- Communication via DDS middleware

## Performance Characteristics

| Component | Frequency | Typical Compute Time |
|-----------|-----------|---------------------|
| MPC Planning | 10 Hz | 50-100 ms |
| PD Tracking | 50 Hz | < 1 ms |
| Safety Filter QP | 50 Hz | 2-5 ms |
| Joint Controller | 50 Hz | < 1 ms |

Total control loop latency: ~7-10 ms

4. API.md (docs/API.md)

text
# API Documentation

## ROS 2 Interfaces

### Topics

#### Published

| Topic | Type | Publisher | Rate | Description |
|-------|------|-----------|------|-------------|
| `/nominal_joint_acceleration` | Float64MultiArray | Nominal Controller | 50 Hz | Desired joint accelerations before safety filtering |
| `/safe_joint_acceleration` | Float64MultiArray | Safety Filter | 50 Hz | Safe joint accelerations after CBF filtering |
| `/goal_reached` | Bool | Nominal Controller | Event | True when end-effector reaches goal |
| `/joint_states` | JointState | Gazebo/Hardware | 50+ Hz | Current robot joint states |
| `/visualization_marker_array` | MarkerArray | Safety Filter | 10 Hz | Obstacle and trajectory visualization |

#### Subscribed

| Topic | Type | Subscriber | Description |
|-------|------|------------|-------------|
| `/joint_states` | JointState | All controllers | Robot state feedback |
| `/nominal_joint_acceleration` | Float64MultiArray | Safety Filter | Nominal commands to filter |
| `/safe_joint_acceleration` | Float64MultiArray | Joint Controller | Filtered safe commands |

### Parameters

#### Safety Filter Node (`safety_filter.py`)

hocbf_controller:
ros__parameters:
# Goal configuration
goal_ee_pos: [0.3, 0.0, 0.5] # Target end-effector position [x, y, z]
goal_tolerance_m: 0.02 # Goal reaching tolerance (meters)
goal_settle_time_s: 1.5 # Time to stay at goal (seconds)

text
# CBF parameters
gamma_js: 10.0                     # CBF gamma parameter (convergence rate)
beta_js: 15.0                      # CBF beta parameter (safety margin)
d_margin: 0.0                      # Additional safety clearance (meters)

# Simulation parameters
max_sim_duration_s: 60.0           # Maximum simulation time
output_data_basename: scenario_0000 # Output file prefix

text

#### MPC Nominal Controller Node

mpc_planner:
ros__parameters:
scenario_config_file: "" # Path to scenario YAML
goal_ee_pos: [0.3, 0.0, 0.5] # Fallback goal if no scenario file
mpc_frequency: 10.0 # MPC replanning rate (Hz)
pid_frequency: 50.0 # PD tracking rate (Hz)
goal_tolerance_m: 0.02 # Goal reaching tolerance
goal_settle_time_s: 2.0 # Settle time at goal

text

#### PD Nominal Controller Node

js_nominal_controller:
ros__parameters:
scenario_config_file: "" # Path to scenario YAML
goal_ee_pos: [0.3, 0.0, 0.5] # Fallback goal
control_frequency: 50.0 # Control loop rate (Hz)
goal_tolerance_m: 0.02 # Goal tolerance
goal_settle_time_s: 2.0 # Settle time

text

## Message Formats

### Float64MultiArray (Joint Accelerations)

std_msgs/Float64MultiArray:
layout:
dim: []
data_offset: 0
data: [ddq1, ddq2, ddq3, ddq4, ddq5, ddq6, ddq7] # rad/s²

text

**Indexing**:
- `ddq1`: fr3_joint1 (shoulder pan)
- `ddq2`: fr3_joint2 (shoulder lift)
- `ddq3`: fr3_joint3 (shoulder roll)
- `ddq4`: fr3_joint4 (elbow)
- `ddq5`: fr3_joint5 (wrist 1)
- `ddq6`: fr3_joint6 (wrist 2)
- `ddq7`: fr3_joint7 (wrist 3)

### JointState

Standard ROS 2 sensor_msgs/JointState with:
- `name`: Joint names (fr3_joint1 through fr3_joint7)
- `position`: Joint angles (radians)
- `velocity`: Joint velocities (rad/s)
- `effort`: Joint torques (N⋅m, usually empty in simulation)

## Python API

### Nominal Controller Interface

To create a custom nominal controller:

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

class MyCustomController(Node):
def init(self):
super().init('my_controller')

text
    # Subscribe to joint states
    self.joint_sub = self.create_subscription(
        JointState,
        '/joint_states',
        self.joint_callback,
        10
    )
    
    # Publish nominal accelerations
    self.accel_pub = self.create_publisher(
        Float64MultiArray,
        '/nominal_joint_acceleration',
        10
    )
    
    # Control loop at 50 Hz
    self.timer = self.create_timer(0.02, self.control_loop)
    
    self.current_q = np.zeros(7)
    self.current_dq = np.zeros(7)

def joint_callback(self, msg):
    # Extract joint positions and velocities
    # (implementation depends on your joint ordering)
    self.current_q = np.array(msg.position[:7])
    self.current_dq = np.array(msg.velocity[:7])

def control_loop(self):
    # Compute your desired acceleration
    ddq_desired = self.compute_control(self.current_q, self.current_dq)
    
    # Publish
    msg = Float64MultiArray()
    msg.data = ddq_desired.tolist()
    self.accel_pub.publish(msg)

def compute_control(self, q, dq):
    # Your control law here
    return np.zeros(7)  # Placeholder

def main(args=None):
rclpy.init(args=args)
node = MyCustomController()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()

if name == 'main':
main()

text

## C++ API

### Joint Position Controller Plugin

To modify or extend the joint position controller:

#include <rclcpp/rclcpp.hpp>
#include <controller_interface/controller_interface.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

namespace cbf_safety_filter {

class JointPositionController : public controller_interface::ControllerInterface {
public:
controller_interface::CallbackReturn on_init() override;
controller_interface::InterfaceConfiguration command_interface_configuration() const override;
controller_interface::InterfaceConfiguration state_interface_configuration() const override;
controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
controller_interface::return_type update(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
void acceleration_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr acceleration_sub_;
std::vector<double> desired_accelerations_;
std::vector<double> current_positions_;
std::vector<double> current_velocities_;
};

} // namespace cbf_safety_filter

text

## Service Interfaces

Currently, the system operates on a topic-based architecture. Future versions may include:
- `/reset_simulation` - Reset robot to initial state
- `/update_obstacles` - Dynamically add/remove obstacles
- `/tune_cbf_parameters` - Online parameter tuning

## Coordinate Frames

- **world**: Fixed world frame (Gazebo origin)
- **fr3_link0**: Robot base frame
- **fr3_link1** through **fr3_link7**: Link frames
- **fr3_hand_tcp**: End-effector (tool center point)

All positions are in meters, orientations in radians.

5. CONTROLLERS.md (docs/CONTROLLERS.md)

text
# Controller Documentation

## Nominal Controllers

Two nominal controller options are provided. Both publish to `/nominal_joint_acceleration` at 50 Hz.

## 1. MPC Nominal Controller

### Overview

Model Predictive Control (MPC) based trajectory planner with PD tracking.

**Files**:
- `mpc_nominal_controller.py`
- Launch: `mpc_nominal_controller.launch.py`

### Architecture

Two-layer control:
1. **High-level MPC** (10 Hz): Plans optimal trajectory
2. **Low-level PD** (50 Hz): Tracks planned trajectory

### MPC Formulation

**Prediction horizon**: 7 steps  
**Time step**: 0.1 seconds (1/mpc_frequency)

**Optimization problem**:

minimize: Σ Q_pos^(k/2) ||ee_pos_k - goal||² + R ||ddq||² + W_jerk ||Δddq||²

subject to: q_{k+1} = q_k + dq_kdt + 0.5ddq_kdt²
dq_{k+1} = dq_k + ddq_kdt
q_min ≤ q_k ≤ q_max
dq_min ≤ dq_k ≤ dq_max
ddq_min ≤ ddq_k ≤ ddq_max
Stoppability constraint

text

**Parameters**:
- Q_pos = 100.0: Position error weight
- R_accel = 0.001: Acceleration regularization
- W_jerk = 0.01: Jerk minimization

**Solver**: CasADi with IPOPT

### PD Tracking

The PD controller tracks the MPC trajectory:

ddq_cmd = Kp*(q_ref - q) + Kd*(dq_ref - dq)

text

**Gains**:
- Kp = 400.0 (position)
- Kd = 40.0 (velocity)

### Goal Reaching

Declares goal reached when:
1. Distance to goal < tolerance (default: 0.02 m)
2. Stayed within tolerance for settle_time (default: 2.0 s)

Upon reaching goal:
- Publishes stationary trajectory (current position, zero velocity)
- Publishes `/goal_reached` = True

### Usage

ros2 launch cbf_safety_filter mpc_nominal_controller.launch.py
scenario_config_file:=path/to/scenario.yaml
mpc_frequency:=10.0
pid_frequency:=50.0

text

### Tuning Guide

| Parameter | Effect | Increase if... | Decrease if... |
|-----------|--------|---------------|----------------|
| Q_pos | Goal attraction | Slow convergence | Oscillations near goal |
| R_accel | Smoothness | Jerky motion | Too slow |
| W_jerk | Jerk penalty | Abrupt changes | Over-smoothed |
| mpc_frequency | Replanning rate | Need faster adaptation | Computational cost too high |

## 2. Joint-Space PD Controller

### Overview

Operational-space PD controller with null-space optimization.

**Files**:
- `pd_nominal_controller.py`
- Launch: `pd_nominal_controller.launch.py`

### Control Law

**Primary task** (operational space):

f_desired = Kp_cart*(goal - ee_pos) + Kd_cart*(0 - ee_vel)
ddq_primary = J^†*f_desired

text

**Secondary task** (null space):

ddq_secondary = Kp_limit * gradient(joint_limit_cost)
ddq_null = (I - J^†J) * ddq_secondary

text

**Combined**:

ddq_cmd = ddq_primary + ddq_null

text

### Parameters

**Operational space gains**:
- Kp_cart = 500.0: Cartesian position gain
- Kd_cart = 50.0: Cartesian velocity gain

**Null-space parameters**:
- Kp_limit = 200.0: Joint limit avoidance gain
- activation_threshold = 0.9: Start avoiding at 90% of range

**Target interpolation**:
- alpha = 0.01: Smooth target filter

### Jacobian Computation

Uses damped pseudo-inverse:

J^† = J^T(JJ^T + λI)^(-1)

text

where λ = 0.01 (damping factor)

### Joint Limit Avoidance

Activates when joint approaches limits (90% of range):

For each joint i:
if |q_i - q_mid| > 0.9 * range/2:
gradient_i = -2*(q_i - q_mid)/range²
ddq_secondary_i = Kp_limit * gradient_i

text

This creates a repulsive potential that keeps joints away from limits.

### Goal Reaching

Same as MPC controller:
- Distance threshold: 0.02 m
- Settle time: 2.0 s

### Usage

ros2 launch cbf_safety_filter pd_nominal_controller.launch.py
scenario_config_file:=path/to/scenario.yaml
control_frequency:=50.0

text

### Tuning Guide

| Parameter | Effect | Increase if... | Decrease if... |
|-----------|--------|---------------|----------------|
| Kp_cart | Position stiffness | Slow response | Oscillations |
| Kd_cart | Damping | Overshoot | Too sluggish |
| Kp_limit | Limit avoidance | Joints hit limits | Interferes with task |
| lambda_damp | Jacobian damping | Numerical issues | Loss of accuracy |

## Comparison

| Feature | MPC Controller | PD Controller |
|---------|---------------|---------------|
| Planning | Yes (7-step horizon) | No (reactive) |
| Obstacle awareness | No (relies on CBF) | No (relies on CBF) |
| Computational cost | Higher (MPC solve) | Lower (analytical) |
| Smoothness | Smoother (jerk penalty) | Good (damped) |
| Goal tracking | Excellent (trajectory planning) | Good (direct control) |
| Joint limit handling | Implicit (constraints) | Explicit (null-space) |
| Suitable for | Complex tasks, dynamic goals | Simple point-to-point |
| Real-time performance | 10 Hz planning + 50 Hz tracking | 50 Hz control |

## Switching Controllers

To switch between controllers during a simulation:

1. Stop current controller: `Ctrl+C` in its terminal
2. Start new controller with same scenario file
3. Safety filter continues operating seamlessly

The controllers are hot-swappable because they share the same interface (`/nominal_joint_acceleration`).

## Creating Custom Controllers

### Requirements

Your controller must:
1. Subscribe to `/joint_states`
2. Publish to `/nominal_joint_acceleration` at ~50 Hz
3. Publish `Float64MultiArray` with 7 elements (joint accelerations)
4. Optionally publish to `/goal_reached` when done

### Template

See [API.md](API.md) for a minimal template.

### Best Practices

1. **Always respect joint limits** in your controller
2. **Smooth outputs**: Avoid discontinuous accelerations
3. **Handle goal proximity**: Reduce commands near goal
4. **Publish regularly**: Don't skip control cycles
5. **Log failures**: Use ROS logging for debugging

6. TROUBLESHOOTING.md (docs/TROUBLESHOOTING.md)

text
# Troubleshooting Guide

## Common Issues

### 1. Controller Not Loading

**Symptom**: Joint position controller fails to spawn

**Solutions**:

Check controller manager is running

ros2 control list_controllers
Verify controller configuration

ros2 param list /controller_manager
Manually load controller

ros2 control load_controller joint_position_controller
ros2 control set_controller_state joint_position_controller active

text

### 2. Safety Filter Not Receiving Accelerations

**Symptom**: "Waiting for nominal acceleration..." message persists

**Check**:

Verify nominal controller is running

ros2 node list | grep controller
Check topic is publishing

ros2 topic hz /nominal_joint_acceleration
Monitor topic content

ros2 topic echo /nominal_joint_acceleration

text

**Solution**: Ensure nominal controller is launched with correct scenario file.

### 3. Robot Not Moving

**Symptom**: Robot spawns but doesn't move toward goal

**Debugging steps**:
1. Check if goal is reachable:

ros2 topic echo /goal_reached

text

2. Verify accelerations are non-zero:

ros2 topic echo /safe_joint_acceleration

text

3. Check for QP solver failures in safety filter logs

4. Inspect obstacle configuration - might be blocking path

**Solution**: 
- Adjust goal position in scenario file
- Tune CBF parameters (reduce gamma/beta for more aggressive motion)
- Remove/relocate obstacles

### 4. QP Solver Fails

**Symptom**: "QP solve failed" in safety filter logs

**Causes**:
- Infeasible constraints (robot surrounded by obstacles)
- Numerical issues (poorly conditioned matrices)
- Joint limits violated

**Solutions**:

In scenario file, try:

d_margin: 0.05 # Increase safety margin
gamma_js: 5.0 # Reduce aggressiveness
beta_js: 10.0 # Reduce conservatism

text

### 5. Gazebo Crashes or Freezes

**Symptom**: Gazebo becomes unresponsive

**Solutions**:

Kill all Gazebo processes

pkill -f "gz sim"
pkill -f gazebo
Clean Gazebo cache

rm -rf ~/.gazebo/
Check GPU drivers if using GUI

nvidia-smi # For NVIDIA GPUs

text

### 6. MPC Solver Timeout

**Symptom**: MPC taking too long (>100ms)

**Solutions**:
1. Reduce horizon length in code (N = 5 instead of 7)
2. Loosen convergence tolerance
3. Simplify cost function
4. Use faster solver (e.g., qpOASES instead of IPOPT)

### 7. Build Errors

**Missing dependencies**:

Install Python packages

pip install numpy scipy cvxpy casadi pinocchio pyyaml matplotlib pandas
Install ROS dependencies

rosdep install --from-paths src --ignore-src -r -y

text

**Pinocchio not found**:

Install Pinocchio

sudo apt install ros-${ROS_DISTRO}-pinocchio

text

**CVX

PY solver issues**:

pip install --upgrade cvxpy
pip install osqp # Recommended solver

text

### 8. Scenario File Not Found

**Symptom**: "Scenario file not found" error

**Solution**:

Use absolute path

ros2 launch cbf_safety_filter gazebo_simulation.launch.py
scenario_config_file:=$(ros2 pkg prefix cbf_safety_filter)/share/cbf_safety_filter/config/generated_scenarios/scenario_0000.yaml
Or cd to install directory

cd ~/ros2_ws/install/cbf_safety_filter/share/cbf_safety_filter/config/generated_scenarios

text

### 9. Visualization Not Working

**Symptom**: RViz shows no robot or obstacles

**Solutions**:

Check RViz subscriptions

ros2 topic list | grep marker
Verify robot description is published

ros2 topic echo /robot_description
Reset RViz config

ros2 launch cbf_safety_filter rviz.launch.py
Then: File → Reload Config

text

### 10. Data Logging Issues

**Symptom**: No CSV files generated after simulation

**Check**:

Verify output directory exists

ls install/cbf_safety_filter/share/cbf_safety_filter/plots/sim/
Check write permissions

ls -l install/cbf_safety_filter/share/cbf_safety_filter/plots/
Look for error messages in safety filter logs

text

**Solution**: Ensure output directories exist and are writable.

## Performance Issues

### Simulation Running Slowly

1. **Use headless mode**:

headless:=True

text

2. **Reduce visualization frequency** in code

3. **Disable logging** for faster execution:
- Comment out pandas logging in safety_filter.py

4. **Use faster Gazebo settings**:
- Reduce physics update rate
- Simplify collision meshes

### High CPU Usage

- Close unnecessary applications
- Reduce MPC frequency to 5 Hz
- Use simpler nominal controller (PD instead of MPC)
- Run batch simulations on server/cluster

## Getting Help

If issues persist:

1. **Check logs**:

ros2 topic echo /rosout

text

2. **Enable debug logging**:

ros2 run cbf_safety_filter safety_filter.py --ros-args --log-level debug

text

3. **Create an issue** on GitHub with:
- ROS 2 distribution
- Error messages
- Scenario file
- Steps to reproduce

4. **Community support**:
- ROS Discourse: https://discourse.ros.org
- ROS Answers: https://answers.ros.org