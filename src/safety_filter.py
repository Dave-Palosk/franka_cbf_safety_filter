#!/usr/bin/env python3
"""
Safety Filter Node - Higher-Order Control Barrier Function (HOCBF) implementation
Ensures collision-free motion for Franka FR3 manipulator by filtering nominal accelerations.
"""

import rclpy
import rclpy.logging
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import pinocchio as pin
import os
import yaml
import pandas as pd
import time as timer

# =============================================================================
# ROBOT MODEL SETUP
# =============================================================================

# Load robot URDF model using Pinocchio for kinematics and dynamics
URDF_FILENAME = "fr3_robot.urdf"
package_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF_PATH = os.path.join(package_directory, "include", "urdf", URDF_FILENAME)

if not os.path.exists(URDF_PATH):
    rclpy.logging.get_logger('cbf_controller_node').error(f"URDF file '{URDF_FILENAME}' not found at {URDF_PATH}.")
try:
    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
    print(f"Pinocchio model loaded from {URDF_PATH}: {model.name}")
except Exception as e:
    rclpy.logging.get_logger('cbf_controller_node').error(f"Error loading URDF with Pinocchio: {e}")
    exit()

NUM_ARM_JOINTS = 7
if model.nv < NUM_ARM_JOINTS:
    rclpy.logging.get_logger('cbf_controller_node').error(f"Error: Model nv ({model.nv}) is less than expected NUM_ARM_JOINTS ({NUM_ARM_JOINTS})")
    exit()

# End-effector frame for trajectory tracking
EE_FRAME_NAME = "fr3_hand_tcp"
try:
    EE_FRAME_ID = model.getFrameId(EE_FRAME_NAME)
except IndexError:
    rclpy.logging.get_logger('cbf_controller_node').error(f"Error: End-effector frame '{EE_FRAME_NAME}' not found in URDF.")
    rclpy.logging.get_logger('cbf_controller_node').error(f"Available frames: {[f.name for f in model.frames]}")
    exit()

# =============================================================================
# COLLISION DETECTION SETUP
# =============================================================================

# Define robot links as capsules (cylinders with spherical caps) for collision checking
# Each link is defined by start/end frames and a radius
links_def = [
    {'name': 'link2_base', 'start_frame_name': 'fr3_link2_offset1', 'end_frame_name': 'fr3_link2_offset2', 'radius': 0.055},
    {'name': 'link2', 'start_frame_name': 'fr3_link2', 'end_frame_name': 'fr3_link3', 'radius': 0.06},
    {'name': 'joint4',   'start_frame_name': 'fr3_link4', 'end_frame_name': 'fr3_link5_offset1', 'radius': 0.065},
    {'name': 'forearm1',   'start_frame_name': 'fr3_link5_offset2', 'end_frame_name': 'fr3_link5_offset3', 'radius': 0.035},
    {'name': 'forearm2',   'start_frame_name': 'fr3_link5_offset3', 'end_frame_name': 'fr3_link5', 'radius': 0.05},
    {'name': 'wrist',     'start_frame_name': 'fr3_link7_offset1', 'end_frame_name': 'fr3_hand',  'radius': 0.055},
    {'name': 'hand',     'start_frame_name': 'fr3_hand_offset1', 'end_frame_name': 'fr3_hand_offset2',  'radius': 0.03},
    {'name': 'end_effector',      'start_frame_name': EE_FRAME_NAME,  'end_frame_name': EE_FRAME_NAME, 'radius': 0.03},
]
active_links = []

# Self-collision pairs to check (links that can collide with each other)
self_collision_link_pair = [
    ('link2', 'hand'),
    ('link2', 'end_effector'),
    ('link2', 'forearm1')
]

# Initialize active links by retrieving frame IDs from the model
rclpy.logging.get_logger('cbf_controller_node').info("Setting up control links (capsules)...")
for link_def in links_def:
    try:
        start_frame_id = model.getFrameId(link_def['start_frame_name'])
        end_frame_id = model.getFrameId(link_def['end_frame_name'])
        active_links.append({
            'name': link_def['name'],
            'start_frame_id': start_frame_id,
            'end_frame_id': end_frame_id,
            'radius': link_def['radius']
        })
    except IndexError as e:
        rclpy.logging.get_logger('cbf_controller_node').warn(
            f"  ! Warning: A frame for link '{link_def['name']}' not found in URDF: {e}. Skipping."
        )

if not active_links:
    rclpy.logging.get_logger('cbf_controller_node').error("No active links defined. Cannot run CBF. Exiting.")
    exit()

# =============================================================================
# JOINT LIMITS
# =============================================================================

# Joint acceleration limits (rad/s²)
ddq_max_scalar = 40.0
ddq_max_arm = np.full(NUM_ARM_JOINTS, ddq_max_scalar)
ddq_min_arm = np.full(NUM_ARM_JOINTS, -ddq_max_scalar)

# Joint velocity limits (rad/s)
dq_max_arm = np.array([2.0, 1.0, 1.5, 1.25, 3.0, 1.5, 3.0])
dq_min_arm = - dq_max_arm

# Joint position limits (rad)
q_max_arm = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
q_min_arm = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])

# =============================================================================
# DYNAMIC CBF PARAMETER ADJUSTMENT CONSTANTS
# =============================================================================

# Constants for automatically adjusting gamma and beta based on current state
EPSILON_DENOMINATOR = 1e-6  # Avoid division by zero
GAMMA_ADJUST_BUFFER = 0.5   # Safety buffer added to minimum required gamma
GAMMA_MAX_LIMIT = 200.0     # Maximum allowed gamma value
BETA_ADJUST_BUFFER = 0.5    # Safety buffer added to minimum required beta
BETA_MAX_LIMIT = 250.0      # Maximum allowed beta value

# =============================================================================
# GEOMETRY HELPER FUNCTIONS
# =============================================================================

def get_closest_points_between_segments(p1, p2, q1, q2):
    """
    Calculates the closest points between two line segments (capsule collision detection).
    
    Args:
        p1, p2: Endpoints of the first segment (robot link)
        q1, q2: Endpoints of the second segment (obstacle or other link)
    
    Returns:
        c1: Closest point on first segment
        c2: Closest point on second segment
        s: Interpolation parameter for first segment [0,1]
        t: Interpolation parameter for second segment [0,1]
    """
    u = p2 - p1
    v = q2 - q1
    w = p1 - q1

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)

    D = a * c - b * b  # Denominator for parameter calculation
    
    s_c, t_c = D, D
    s, t = 0.0, 0.0

    # Handle parallel segments
    if D < 1e-7:
        s = np.clip(-d / a, 0.0, 1.0) if a > 1e-7 else 0.0
        t = (b * s + e) / c if c > 1e-7 else 0.0
    else:
        # Compute closest points on infinite lines, then clamp to segments
        s_c = (b * e - c * d)
        t_c = (a * e - b * d)
        
        if s_c < 0.0:
            s_c = 0.0
            t_c = e
            t_c_denom = c
        elif s_c > D:
            s_c = D
            t_c = e + b
            t_c_denom = c
        else:
            t_c_denom = D
        
        s = s_c / D
        
        if t_c < 0.0:
            t = 0.0
            s_num = -d
            if s_num < 0.0:
                s = 0.0
            elif s_num > a:
                s = 1.0
            else:
                s = s_num / a if a > 1e-7 else 0.0
        elif t_c > t_c_denom:
            t = 1.0
            s_num = -d + b
            if s_num < 0.0:
                s = 0.0
            elif s_num > a:
                s = 1.0
            else:
                s = s_num / a if a > 1e-7 else 0.0
        else:
            t = t_c / t_c_denom

    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)

    c1 = p1 + s * u
    c2 = q1 + t * v
    
    return c1, c2, s, t

# =============================================================================
# HIGHER-ORDER CONTROL BARRIER FUNCTION (HOCBF) FORMULATION
# =============================================================================

def h_func(p_rel, robot_link_radius, obstacle_radius, d_margin=0):
    """
    Barrier function h(x): measures safety distance between capsules.
    h > 0: safe, h = 0: boundary, h < 0: collision
    
    Args:
        p_rel: Relative position vector (robot point - obstacle point)
        robot_link_radius: Radius of robot link capsule
        obstacle_radius: Radius of obstacle capsule
        d_margin: Additional safety margin
    
    Returns:
        h value (safe if positive)
    """
    R_eff_sq = (robot_link_radius + obstacle_radius + d_margin)**2
    return np.dot(p_rel, p_rel) - R_eff_sq

def Lf_h(p_rel, v_rel):
    """
    Lie derivative of h along drift dynamics (time derivative of h).
    
    Args:
        p_rel: Relative position vector
        v_rel: Relative velocity vector (robot - obstacle)
    
    Returns:
        Time derivative of h
    """
    return 2 * np.dot(p_rel, v_rel)

def psi_func(h_val, Lf_h_val, gamma_param):
    """
    Psi function for HOCBF: ψ = ḣ + γ*h
    This must remain positive for safety.
    
    Args:
        h_val: Current barrier function value
        Lf_h_val: Time derivative of h
        gamma_param: CBF convergence rate parameter
    
    Returns:
        Psi value
    """
    return Lf_h_val + gamma_param * h_val

def Lf_psi(v_rel, Lf_h_val, gamma_param):
    """
    Lie derivative of psi along drift dynamics (time derivative of ψ).
    
    Args:
        v_rel: Relative velocity vector
        Lf_h_val: Time derivative of h
        gamma_param: CBF gamma parameter
    
    Returns:
        Time derivative of psi
    """
    term_vel_sq = 2 * np.dot(v_rel, v_rel)
    return term_vel_sq + gamma_param * Lf_h_val

def Lg_psi(J_p_robot_closest, p_rel):
    """
    Lie derivative of psi along control input (how control affects ψ).
    
    Args:
        J_p_robot_closest: Jacobian at closest point on robot link
        p_rel: Relative position vector
    
    Returns:
        Vector showing control influence on psi
    """
    J_p_arm = J_p_robot_closest[:, :NUM_ARM_JOINTS]
    return 2 * np.dot(p_rel, J_p_arm)

# =============================================================================
# KINEMATICS HELPER
# =============================================================================

def get_point_kinematics(start_frame_id, end_frame_id, t, dq_full_curr):
    """
    Compute Jacobian and velocity at interpolated point on a robot link.
    
    Args:
        start_frame_id: Frame ID of link start
        end_frame_id: Frame ID of link end
        t: Interpolation parameter [0,1]
        dq_full_curr: Current joint velocities
    
    Returns:
        J_C: Jacobian at point C
        v_C: Velocity at point C
    """
    # Get Jacobians at both ends of the link
    J_full_p1 = pin.getFrameJacobian(model, data, start_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
    J_full_p2 = pin.getFrameJacobian(model, data, end_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
    
    # Interpolate Jacobian
    J_C = (1 - t) * J_full_p1 + t * J_full_p2

    # Compute velocity at interpolated point
    v_C = J_C @ dq_full_curr
    return J_C, v_C

# =============================================================================
# SAFETY FILTER - HOCBF QUADRATIC PROGRAM (QP) SOLVER
# =============================================================================

def solve_hocbf_qp(dt_val, q_full_curr, dq_full_curr, ddq_nominal, current_gamma_js_val, current_beta_js_val, d_margin, current_obstacles_list_sim, active_links_list, node_logger):
    """
    Solves the HOCBF-QP to find safe joint accelerations.
    
    Optimization problem:
        minimize: ||u - u_nominal||²
        subject to: CBF constraints (ensure h and ψ remain positive)
                    Joint limits (position, velocity, acceleration)
    
    Args:
        dt_val: Control timestep
        q_full_curr: Current joint positions
        dq_full_curr: Current joint velocities
        ddq_nominal: Nominal (desired) joint accelerations from controller
        current_gamma_js_val: Current gamma parameter
        current_beta_js_val: Current beta parameter
        d_margin: Additional safety margin
        current_obstacles_list_sim: List of obstacles
        active_links_list: List of robot links to check
        node_logger: ROS logger for warnings
    
    Returns:
        ddq_safe: Safe joint accelerations
        qp_solved: Boolean indicating if QP was successfully solved
    """
    # Decision variable: safe joint accelerations
    u_qp_js = cp.Variable(NUM_ARM_JOINTS)

    ddq_nominal_np = np.array(ddq_nominal).flatten()

    # Cost function: minimize deviation from nominal controller
    cost = cp.sum_squares(u_qp_js - ddq_nominal_np)
    constraints_qp = []
    qp_failed_flag = False

    # Pre-calculate kinematics once (efficient for multiple constraint generation)
    ddq_full_zeros = np.zeros(model.nv)
    pin.forwardKinematics(model, data, q_full_curr, dq_full_curr, ddq_full_zeros)
    pin.computeJointJacobians(model, data, q_full_curr)
    pin.updateFramePlacements(model, data)

    # Generate HOCBF constraints for each robot link vs each obstacle
    for link_info in active_links_list:
        start_frame_id = link_info['start_frame_id']
        end_frame_id = link_info['end_frame_id']
        link_radius_val = link_info['radius']

        # Get link endpoints in world frame
        p1 = data.oMf[start_frame_id].translation
        p2 = data.oMf[end_frame_id].translation

        for obs_item in current_obstacles_list_sim:
            # Find closest points between robot link and obstacle
            c_robot, c_obs, t1, t2 = get_closest_points_between_segments(
                p1, p2, obs_item['pose_start'], obs_item['pose_end']
            )
            
            p_rel = c_robot - c_obs
            h_val = h_func(p_rel, link_radius_val, obs_item['radius'], d_margin)
            
            # Only add constraint if h is small (potential collision risk)
            if h_val < 0.07:
                # Compute kinematics at closest point on robot
                J_C, v_C = get_point_kinematics(
                    link_info['start_frame_id'], link_info['end_frame_id'], t1, dq_full_curr
                )

                # Relative velocity (assuming obstacle has constant velocity)
                v_rel = v_C - obs_item['velocity']

                # Compute HOCBF terms
                Lf_h_val = Lf_h(p_rel, v_rel)
                psi_val = psi_func(h_val, Lf_h_val, current_gamma_js_val)
                Lf_psi_val = Lf_psi(v_rel, Lf_h_val, current_gamma_js_val)
                Lg_psi_val_arm = Lg_psi(J_C, p_rel)

                # Add HOCBF constraint: Lg_psi * u >= -Lf_psi - beta * psi
                constraints_qp.append(Lg_psi_val_arm @ u_qp_js >= -Lf_psi_val - current_beta_js_val * psi_val)

    # Generate self-collision constraints
    for link1, link2 in self_collision_link_pair:
        link1_info = next((l for l in active_links_list if l['name'] == link1), None)
        start_frame_id_1 = link1_info['start_frame_id']
        end_frame_id_1 = link1_info['end_frame_id']
        link_radius_val_1 = link1_info['radius']
        p1_1 = data.oMf[start_frame_id_1].translation
        p2_1 = data.oMf[end_frame_id_1].translation

        link2_info = next((l for l in active_links_list if l['name'] == link2), None)
        start_frame_id_2 = link2_info['start_frame_id']
        end_frame_id_2 = link2_info['end_frame_id']
        link_radius_val_2 = link2_info['radius']
        p1_2 = data.oMf[start_frame_id_2].translation
        p2_2 = data.oMf[end_frame_id_2].translation

        # Find closest points between the two robot links
        c_link1, c_link2, t1, t2 = get_closest_points_between_segments(p1_1, p2_1, p1_2, p2_2)
        p_rel = c_link1 - c_link2
        h_val = h_func(p_rel, link_radius_val_1, link_radius_val_2)
        
        # Only add constraint if links are close
        if h_val < 0.03:
            J_C1, v_C1 = get_point_kinematics(start_frame_id_1, end_frame_id_1, t1, dq_full_curr)
            J_C2, v_C2 = get_point_kinematics(start_frame_id_2, end_frame_id_2, t2, dq_full_curr)
            
            v_rel = v_C1 - v_C2
            J_rel = J_C1 - J_C2

            Lf_h_val = Lf_h(p_rel, v_rel)
            psi_val = psi_func(h_val, Lf_h_val, current_gamma_js_val)
            Lf_psi_val = Lf_psi(v_rel, Lf_h_val, current_gamma_js_val)
            Lg_psi_val_arm = Lg_psi(J_rel, p_rel)

            constraints_qp.append(Lg_psi_val_arm @ u_qp_js >= -Lf_psi_val - current_beta_js_val * psi_val)

    # Add joint limit constraints
    # Acceleration limits
    constraints_qp.append(u_qp_js >= ddq_min_arm)
    constraints_qp.append(u_qp_js <= ddq_max_arm)

    # Velocity limits (enforced via forward integration)
    dq_current_arm_val = dq_full_curr[:NUM_ARM_JOINTS]
    constraints_qp.append(u_qp_js >= (dq_min_arm - dq_current_arm_val) / dt_val)
    constraints_qp.append(u_qp_js <= (dq_max_arm - dq_current_arm_val) / dt_val)

    # Position limits (enforced via double integration)
    q_current_arm_val = q_full_curr[:NUM_ARM_JOINTS]
    constraints_qp.append(u_qp_js >= 2 * (q_min_arm - q_current_arm_val - dq_current_arm_val * dt_val) / (dt_val**2))
    constraints_qp.append(u_qp_js <= 2 * (q_max_arm - q_current_arm_val - dq_current_arm_val * dt_val) / (dt_val**2))

    # Solve the QP
    problem = cp.Problem(cp.Minimize(cost), constraints_qp)
    ddq_safe_arm_val = np.zeros(NUM_ARM_JOINTS)
    qp_solved_successfully = False 

    try:
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=25000)
    except cp.error.SolverError:
        node_logger.warn(f"OSQP SolverError. Trying default solver.")
        try:
            problem.solve(verbose=False)
        except Exception as e_alt_solve:
            qp_failed_flag = True
            node_logger.warn("OSQP SolverError. Attempting emergency stop.") 

    # Check solution status
    if not qp_failed_flag and (problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE):
        if u_qp_js.value is not None:
            ddq_safe_arm_val = u_qp_js.value
            qp_solved_successfully = True
        else: 
            qp_failed_flag = True
            node_logger.warn("QP solved, but u_qp_js.value is None. Emergency stop activated.")
    else: 
        qp_failed_flag = True
        node_logger.warn(f"QP status {problem.status}. Emergency stop activated.")
    
    return np.array(ddq_safe_arm_val).flatten(), qp_solved_successfully

# =============================================================================
# ROS 2 NODE - CBF SAFETY FILTER
# =============================================================================

class CbfControllerNode(Node):
    """
    ROS 2 node that implements the HOCBF safety filter.
    
    Subscribes to:
        - /joint_states: Current robot state
        - /nominal_joint_acceleration: Desired accelerations from controller
    
    Publishes to:
        - /joint_position_controller/external_commands: Safe joint positions
        - Various marker topics for RViz visualization
    """
    
    def __init__(self):
        super().__init__('cbf_controller_node')
        self.get_logger().info('CBF Controller Node has been started.')

        # Load configuration from scenario YAML file
        self.declare_parameter('scenario_config_file', '')
        scenario_path = self.get_parameter('scenario_config_file').get_parameter_value().string_value
        
        if not scenario_path or not os.path.exists(scenario_path):
            self.get_logger().fatal("HOCBF Node requires 'scenario_config_file', but it was not provided or file does not exist. Shutting down.")
            self.destroy_node()
            return
            
        with open(scenario_path, 'r') as file:
            config = yaml.safe_load(file)
            
        hocbf_params = config.get('hocbf_controller', {}).get('ros__parameters', {})
        obstacle_params_from_config = config.get('obstacles', [])
        
        # Load CBF parameters from scenario file
        self.goal_ee_pos = np.array(hocbf_params.get('goal_ee_pos', [0.3, 0.0, 0.5]))
        self.initial_gamma = float(hocbf_params.get('gamma_js', 2.0))
        self.initial_beta = float(hocbf_params.get('beta_js', 3.0))
        self.d_margin = float(hocbf_params.get('d_margin', 0.0))
        self.output_basename = hocbf_params.get('output_data_basename', 'unnamed_scenario')
        self.initial_ee_pos = None

        # Load termination criteria
        self.goal_tolerance = float(hocbf_params.get('goal_tolerance_m', 0.02))
        self.goal_settle_time_s = float(hocbf_params.get('goal_settle_time_s', 2.0))
        self.max_sim_duration_s = float(hocbf_params.get('max_sim_duration_s', 60.0))
        self.start_time_ns = self.get_clock().now().nanoseconds
        self.at_goal_timer = 0.0
        self.is_shutdown_initiated = False
        self.final_run_status = "NONE"

        self.get_logger().info(f"--- HOCBF Controller Config for '{self.output_basename}' ---")
        
        # Parse obstacle definitions from scenario file
        self.current_obstacles_list_sim = []
        for obs in obstacle_params_from_config:
            self.current_obstacles_list_sim.append({
                'pose_start': np.array(obs['pose_start']['position']),
                'pose_end': np.array(obs['pose_start']['position']),  # Default to sphere
                'radius': float(obs['size']['radius']),
                'velocity': np.array(obs['velocity']['linear'])
            })
            # Update pose_end if obstacle is a capsule (not a sphere)
            if 'pose_end' in obs:
                self.current_obstacles_list_sim[-1]['pose_end'] = np.array(obs['pose_end']['position'])

        # ROS 2 subscriptions
        self.joint_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )
        self.joint_state_subscription

        self.nominal_acceleration_subscription = self.create_subscription(
            Float64MultiArray,
            '/nominal_joint_acceleration',
            self.nominal_acceleration_callback,
            10
        )
        self.nominal_acceleration_subscription

        self.latest_nominal_acceleration = None

        # ROS 2 publishers
        self.joint_command_publisher = self.create_publisher(
            Float64MultiArray,
            '/joint_position_controller/external_commands',
            10
        )

        # Publishers for RViz visualization markers
        # (Multiple publishers needed due to RViz display limitations)
        self.marker_publisher_1 = self.create_publisher(Marker,'/visualization_marker_1', 10)
        self.marker_publisher_2 = self.create_publisher(Marker,'/visualization_marker_2', 10)
        self.marker_publisher_3 = self.create_publisher(Marker,'/visualization_marker_3', 10)
        self.marker_publisher_4 = self.create_publisher(Marker,'/visualization_marker_4', 10)

        self.bot_marker_publisher_1 = self.create_publisher(Marker,'/bot_marker_1', 10)
        self.bot_marker_publisher_2 = self.create_publisher(Marker,'/bot_marker_2', 10)
        self.bot_marker_publisher_3 = self.create_publisher(Marker,'/bot_marker_3', 10)
        self.bot_marker_publisher_4 = self.create_publisher(Marker,'/bot_marker_4', 10)
        self.bot_marker_publisher_5 = self.create_publisher(Marker,'/bot_marker_5', 10)

        self.trajectory_marker_publisher = self.create_publisher(Marker,'/ee_trajectory_marker', 10)

        self.robot_joint_names = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
            'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
        ]
        self.num_arm_joints = len(self.robot_joint_names)

        # Data logging for plotting
        self.history_data_frames = []
        self.ee_position_history = []

        # Dynamic CBF parameters (adjusted in real-time based on state)
        self.current_gamma_js = self.initial_gamma
        self.current_beta_js = self.initial_beta

        # Robot state variables
        self.current_joint_positions = np.zeros(self.num_arm_joints)
        self.current_joint_velocities = np.zeros(self.num_arm_joints)
        self.received_first_joint_state = False

        # Pinocchio state vectors (includes all joints, not just arm)
        self.q_full_pin = np.zeros(model.nq)
        self.dq_full_pin = np.zeros(model.nv)

        # Control loop timing
        self.frequency = 50  # Hz
        self.dt = 1.0 / self.frequency

        # Create timers for periodic callbacks
        self.pid_timer = self.create_timer(self.dt, self.safety_filter_loop)
        self.marker_publish_timer = self.create_timer(0.5, self.publish_markers)

    def joint_states_callback(self, msg):
        """
        Callback for /joint_states topic.
        Updates current robot state.
        """
        if not self.received_first_joint_state:
            # Verify all required joints are present
            for i, name in enumerate(self.robot_joint_names):
                if name not in msg.name:
                    self.get_logger().error(f"Joint '{name}' not found in received joint states. Cannot proceed.")
                    self.destroy_node()
                    return
            self.received_first_joint_state = True
            
        current_positions = []
        current_velocities = []

        # Extract joint data in correct order
        for joint_name in self.robot_joint_names:
            try:
                idx = msg.name.index(joint_name)
                current_positions.append(msg.position[idx])
                current_velocities.append(msg.velocity[idx] if idx < len(msg.velocity) else 0.0)
            except ValueError:
                self.get_logger().warn(f"Joint '{joint_name}' not found in current JointState message. This should not happen if initial check passed.")
                return

        self.current_joint_positions = np.array(current_positions)
        self.current_joint_velocities = np.array(current_velocities)
        
        # Update Pinocchio state
        self.q_full_pin[:self.num_arm_joints] = self.current_joint_positions
        self.dq_full_pin[:self.num_arm_joints] = self.current_joint_velocities
    
    def nominal_acceleration_callback(self, msg):
        """
        Callback for /nominal_joint_acceleration topic.
        Stores nominal accelerations from controller.
        """
        if len(msg.data) == NUM_ARM_JOINTS:
            self.latest_nominal_acceleration = np.array(msg.data)

    def safety_filter_loop(self):
        """
        Main control loop running at 50 Hz.
        Applies HOCBF safety filter to nominal accelerations.
        """
        if not self.received_first_joint_state or self.is_shutdown_initiated:
            return
        
        if self.latest_nominal_acceleration is None:
            self.get_logger().info('Waiting for nominal acceleration...', throttle_duration_sec=1.0)
            return
        
        # Update obstacle positions (for moving obstacles)
        for obs in self.current_obstacles_list_sim:
            obs['pose_start'] = obs['pose_start'] + obs['velocity'] * self.dt

        q_arm_current = self.current_joint_positions
        dq_arm_current = self.current_joint_velocities
        ddq_nominal = self.latest_nominal_acceleration

        # Initialize tracking variables for minimum values across all constraints
        min_h_current = float('inf')
        min_psi_current = float('inf')
        min_dist_current = float('inf')

        max_gamma_required_for_h = -float('inf')
        max_beta_required_for_psi = -float('inf')

        # Update Pinocchio kinematics
        pin.forwardKinematics(model, data, self.q_full_pin, self.dq_full_pin, np.zeros(model.nv))
        pin.updateFramePlacements(model, data)

        # Record end-effector position for trajectory visualization
        current_ee_pos = data.oMf[EE_FRAME_ID].translation
        self.ee_position_history.append(Point(x=current_ee_pos[0], y=current_ee_pos[1], z=current_ee_pos[2]))

        # Calculate state-dependent acceleration bounds (for dynamic parameter adjustment)
        ddq_max_from_vel = (dq_max_arm - dq_arm_current) / self.dt
        ddq_min_from_vel = (dq_min_arm - dq_arm_current) / self.dt
        ddq_max_from_pos = 2 * (q_max_arm - q_arm_current - dq_arm_current * self.dt) / (self.dt**2)
        ddq_min_from_pos = 2 * (q_min_arm - q_arm_current - dq_arm_current * self.dt) / (self.dt**2)
        ddq_max_kinematic = np.minimum(ddq_max_arm, np.minimum(ddq_max_from_vel, ddq_max_from_pos))
        ddq_min_kinematic = np.maximum(ddq_min_arm, np.maximum(ddq_min_from_vel, ddq_min_from_pos))
        
        # First pass: Calculate required gamma for all link-obstacle pairs
        current_h_psi_values = {}

        for link_info in active_links:
            start_frame_id = link_info['start_frame_id']
            end_frame_id = link_info['end_frame_id']
            link_radius_val = link_info['radius']

            p1 = data.oMf[start_frame_id].translation
            p2 = data.oMf[end_frame_id].translation

            for obs_idx, obs_item in enumerate(self.current_obstacles_list_sim):
                c_robot, c_obs, t1, t2 = get_closest_points_between_segments(
                    p1, p2, obs_item['pose_start'], obs_item['pose_end']
                )
                
                J_C, v_C = get_point_kinematics(
                    link_info['start_frame_id'], link_info['end_frame_id'], t1, self.dq_full_pin
                )

                dist_centers = np.linalg.norm(c_robot - c_obs)
                dist_surfaces = dist_centers - (link_radius_val + obs_item['radius'])
                min_dist_current = min(min_dist_current, dist_surfaces)

                p_rel = c_robot - c_obs
                h_val = h_func(p_rel, link_radius_val, obs_item['radius'], self.d_margin)
                min_h_current = min(min_h_current, h_val)
                
                v_rel = v_C - obs_item['velocity']
                Lf_h_val = Lf_h(p_rel, v_rel)

                # Calculate minimum required gamma for this pair
                if h_val > EPSILON_DENOMINATOR:
                    gamma_needed_for_pair = -Lf_h_val / h_val
                    max_gamma_required_for_h = max(max_gamma_required_for_h, gamma_needed_for_pair)
                elif h_val < -EPSILON_DENOMINATOR:
                    max_gamma_required_for_h = GAMMA_MAX_LIMIT

        # Adjust gamma based on requirements
        if max_gamma_required_for_h > -float('inf'):
            adjusted_gamma = max(0.0, max_gamma_required_for_h + GAMMA_ADJUST_BUFFER)
            self.current_gamma_js = max(self.initial_gamma, min(adjusted_gamma, GAMMA_MAX_LIMIT))

        # Second pass: Calculate required beta using updated gamma
        for link_info in active_links:
            start_frame_id = link_info['start_frame_id']
            end_frame_id = link_info['end_frame_id']
            link_radius_val = link_info['radius']

            p1 = data.oMf[start_frame_id].translation
            p2 = data.oMf[end_frame_id].translation

            for obs_idx, obs_item in enumerate(self.current_obstacles_list_sim):
                c_robot, c_obs, t1, t2 = get_closest_points_between_segments(
                    p1, p2, obs_item['pose_start'], obs_item['pose_end']
                )

                J_C, v_C = get_point_kinematics(
                    link_info['start_frame_id'], link_info['end_frame_id'], t1, self.dq_full_pin
                )

                p_rel = c_robot - c_obs
                v_rel = v_C - obs_item['velocity']

                Lf_h_val = Lf_h(p_rel, v_rel)
                psi_val = psi_func(h_func(p_rel, link_radius_val, obs_item['radius'], self.d_margin), Lf_h_val, self.current_gamma_js)
                
                min_psi_current = min(min_psi_current, psi_val)
                
                Lf_psi_val = Lf_psi(v_rel, Lf_h_val, self.current_gamma_js)
                Lg_psi_val = Lg_psi(J_C, p_rel).flatten()
                
                # Calculate supremum of control influence
                sup_Lg_psi_u = np.sum(np.where(
                    Lg_psi_val >= 0, 
                    Lg_psi_val * ddq_max_kinematic, 
                    Lg_psi_val * ddq_min_kinematic))
                
                S_sup_val = Lf_psi_val + sup_Lg_psi_u
                
                # Calculate minimum required beta for this pair
                if psi_val > EPSILON_DENOMINATOR:
                    beta_needed_for_pair = -S_sup_val / psi_val
                    max_beta_required_for_psi = max(max_beta_required_for_psi, beta_needed_for_pair)
                elif psi_val < -EPSILON_DENOMINATOR:
                    max_beta_required_for_psi = BETA_MAX_LIMIT

                # Store values for logging
                key = f"{link_info['name']}_obs{obs_idx}"
                current_h_psi_values[f'h_{key}'] = h_func(p_rel, link_radius_val, obs_item['radius'], self.d_margin)
                current_h_psi_values[f'psi_{key}'] = psi_val

        # Adjust beta based on requirements
        if max_beta_required_for_psi > -float('inf'):
            adjusted_beta = max(0.0, max_beta_required_for_psi + BETA_ADJUST_BUFFER)
            self.current_beta_js = max(self.initial_beta, min(adjusted_beta, BETA_MAX_LIMIT))

        # Get current simulation time
        current_time_ns = self.get_clock().now().nanoseconds
        current_time_s = (current_time_ns - self.start_time_ns) / 1e9

        # Check termination conditions
        current_ee_pos = data.oMf[EE_FRAME_ID].translation
        distance_to_goal = np.linalg.norm(current_ee_pos - self.goal_ee_pos)

        if self.initial_ee_pos is None:
            self.initial_ee_pos = current_ee_pos.copy()

        # Condition 1: Goal reached and settled
        if distance_to_goal < self.goal_tolerance:
            self.at_goal_timer += self.dt
            if self.at_goal_timer >= self.goal_settle_time_s:
                self.get_logger().info(f"SUCCESS: Goal reached for {self.goal_settle_time_s}s. Simulation time {current_time_s:.2f}s.")
                self.final_run_status = "SUCCESS"
                self.is_shutdown_initiated = True
                self.destroy_node()
                return
        else:
            self.at_goal_timer = 0.0

        # Condition 2: Timeout
        if current_time_s > self.max_sim_duration_s:
            self.get_logger().warn(f"TIMEOUT: Simulation exceeded {self.max_sim_duration_s}s. Shutting down.")
            if np.linalg.norm(current_ee_pos - self.initial_ee_pos) < 0.01:
                self.get_logger().error("CONTROLLER FAILED: Robot did not move from initial position.")
                self.final_run_status = "FAILED_NO_MOVEMENT"
            else:
                self.final_run_status = "TIMEOUT"
            self.is_shutdown_initiated = True
            self.destroy_node()
            return
        
        start_solve_time = timer.time()
        
        # Solve HOCBF-QP to get safe accelerations
        ddq_safe_arm_cmd, qp_solved = solve_hocbf_qp(
            self.dt, self.q_full_pin, self.dq_full_pin, ddq_nominal, self.current_gamma_js,
            self.current_beta_js, self.d_margin, self.current_obstacles_list_sim, active_links, self.get_logger()
        )
        
        # Uncomment to bypass safety filter (test nominal controller)
        # qp_solved = True
        # ddq_safe_arm_cmd = ddq_nominal
        
        if not qp_solved:
            # QP infeasible: hold current position
            next_dq_arm = dq_arm_current
            next_q_arm = q_arm_current
            self.get_logger().warn(f"QP infeasible, using current positions: {np.round(q_arm_current, 3)}")
        else:
            # Integrate accelerations to get next state (Euler integration)
            next_dq_arm = dq_arm_current + ddq_safe_arm_cmd * self.dt
            next_q_arm = q_arm_current + dq_arm_current * self.dt + 0.5 * ddq_safe_arm_cmd * self.dt**2

        # Log data for plotting
        step_data = {
            'time': current_time_s,
            'solve_time': timer.time() - start_solve_time,
            'min_h': min_h_current,
            'min_dist': min_dist_current,
            'min_psi': min_psi_current,
            'qp_infeasible': not qp_solved,
            'joint_q': q_arm_current.tolist(),
            'next_q': next_q_arm.tolist(),
            'joint_dq': dq_arm_current.tolist(),
            'next_dq': next_dq_arm.tolist(),
            'joint_ddq': ddq_safe_arm_cmd.tolist(),
            'joint_ddq_nominal': ddq_nominal.tolist(),
            'current_gamma_js': self.current_gamma_js,
            'max_gamma_required_for_h': max_gamma_required_for_h,
            'current_beta_js': self.current_beta_js,
            'max_beta_required_for_psi': max_beta_required_for_psi,
        }
        step_data.update(current_h_psi_values)

        self.history_data_frames.append(step_data)

        # Publish safe joint commands
        self.publish_joint_commands(next_q_arm)

    def publish_joint_commands(self, positions_array):
        """Publish joint position commands to the C++ joint controller."""
        msg = Float64MultiArray()
        msg.data = positions_array.tolist()
        self.joint_command_publisher.publish(msg)

    # =========================================================================
    # VISUALIZATION MARKER CREATION
    # =========================================================================

    def create_sphere_marker(self, marker_id, position, radius, r, g, b, a, frame_id="world"):
        """Create a sphere marker for RViz visualization."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "cbf_viz"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(radius * 2)
        marker.scale.y = float(radius * 2)
        marker.scale.z = float(radius * 2)
        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = float(a)
        marker.lifetime = rclpy.duration.Duration(seconds=0.7).to_msg()

        return marker
    
    def create_cylinder_marker(self, marker_id, p1, p2, radius, r, g, b, a, frame_id="world"):
        """Create a cylinder marker connecting two points (for capsule visualization)."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "cbf_viz_links"
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        # Position at midpoint
        midpoint = (p1 + p2) / 2
        marker.pose.position.x = float(midpoint[0])
        marker.pose.position.y = float(midpoint[1])
        marker.pose.position.z = float(midpoint[2])

        # Scale: diameter and length
        link_vec = p2 - p1
        length = np.linalg.norm(link_vec)
        marker.scale.x = float(radius * 2)
        marker.scale.y = float(radius * 2)
        marker.scale.z = float(length)

        # Orientation: align cylinder Z-axis with link vector
        if length > 1e-6:
            z_axis = np.array([0.0, 0.0, 1.0])
            axis = np.cross(z_axis, link_vec)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-6:
                axis /= axis_norm
                angle = np.arccos(np.dot(z_axis, link_vec) / length)
                q = pin.Quaternion(np.cos(angle / 2.0),
                                   axis[0] * np.sin(angle / 2.0),
                                   axis[1] * np.sin(angle / 2.0),
                                   axis[2] * np.sin(angle / 2.0))
                marker.pose.orientation.w = q.w
                marker.pose.orientation.x = q.x
                marker.pose.orientation.y = q.y
                marker.pose.orientation.z = q.z
            else:
                marker.pose.orientation.w = 1.0
        else:
            marker.pose.orientation.w = 1.0

        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = float(a)
        marker.lifetime = rclpy.duration.Duration(seconds=0.7).to_msg()
        return marker

    def create_goal_marker(self, marker_id, position, frame_id="world"):
        """Create a cube marker for the goal position."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "cbf_viz"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        marker.lifetime = rclpy.duration.Duration(seconds=0.7).to_msg()

        return marker

    def publish_markers(self):
        """Publish visualization markers for RViz (obstacles, robot links, goal, trajectory)."""
        # Publish obstacle capsule markers
        for i, obs in enumerate(self.current_obstacles_list_sim):
            marker_obs_cap1 = self.create_sphere_marker(
                marker_id=i,
                position=obs['pose_start'],
                radius=obs['radius'],
                r=1.0, g=0.5, b=0.0, a=0.7
            )
            marker_obs_cap2 = self.create_sphere_marker(
                marker_id=i+len(self.current_obstacles_list_sim),
                position=obs['pose_end'],
                radius=obs['radius'],
                r=1.0, g=0.5, b=0.0, a=0.7
            )
            marker_obs_cyl = self.create_cylinder_marker(
                marker_id=i+len(self.current_obstacles_list_sim)*2,
                p1=obs['pose_start'],
                p2=obs['pose_end'],
                radius=obs['radius'],
                r=1.0, g=0.5, b=0.0, a=0.7
            )
            self.marker_publisher_2.publish(marker_obs_cap1)
            if not np.array_equal(obs['pose_start'], obs['pose_end']):
                self.marker_publisher_3.publish(marker_obs_cap2)
                self.marker_publisher_4.publish(marker_obs_cyl)

        # Publish goal position marker
        marker_goal = self.create_goal_marker(
            marker_id=len(self.current_obstacles_list_sim)*3,
            position=self.goal_ee_pos
        )
        self.marker_publisher_1.publish(marker_goal)
        
        base_marker_id_link = len(self.current_obstacles_list_sim)*3 + 1 

        # Update kinematics for current robot pose
        pin.forwardKinematics(model, data, self.q_full_pin, self.dq_full_pin, np.zeros(model.nv))
        pin.updateFramePlacements(model, data)

        # Publish robot link capsule markers
        # (Distributed across multiple publishers due to RViz limitations)
        bot_number_marker = 0
        for i, link_info in enumerate(active_links):
            p1 = data.oMf[link_info['start_frame_id']].translation
            p2 = data.oMf[link_info['end_frame_id']].translation
            
            marker_link = self.create_cylinder_marker(
                marker_id=base_marker_id_link + i,
                p1=p1,
                p2=p2,
                radius=link_info['radius'],
                r=0.2, g=0.2, b=0.8, a=0.6
            )
            marker_cap_1 = self.create_sphere_marker(
                marker_id=base_marker_id_link + len(active_links) + i,
                position=p1,
                radius=link_info['radius'],
                r=0.2, g=0.2, b=0.8, a=0.6
            )
            marker_cap_2 = self.create_sphere_marker(
                marker_id=base_marker_id_link + 2*len(active_links) + i,
                position=p2,
                radius=link_info['radius'],
                r=0.2, g=0.2, b=0.8, a=0.6
            )

            # Distribute markers among publishers (max 5 per publisher for RViz)
            if bot_number_marker < 5:
                self.bot_marker_publisher_1.publish(marker_cap_1)
                bot_number_marker += 1
                if p1.tolist() != p2.tolist():
                    if bot_number_marker < 5:
                        self.bot_marker_publisher_1.publish(marker_cap_2)
                        bot_number_marker += 1
                    else:
                        self.bot_marker_publisher_2.publish(marker_cap_2)
                        bot_number_marker += 1
                    if bot_number_marker < 5:
                        self.bot_marker_publisher_1.publish(marker_link)
                        bot_number_marker += 1
                    else:
                        self.bot_marker_publisher_2.publish(marker_link)
                        bot_number_marker += 1
            elif bot_number_marker < 10:
                self.bot_marker_publisher_2.publish(marker_cap_1)
                bot_number_marker += 1
                if p1.tolist() != p2.tolist():
                    if bot_number_marker < 10:
                        self.bot_marker_publisher_2.publish(marker_cap_2)
                        bot_number_marker += 1
                    else:
                        self.bot_marker_publisher_3.publish(marker_cap_2)
                        bot_number_marker += 1
                    if bot_number_marker < 10:
                        self.bot_marker_publisher_2.publish(marker_link)
                        bot_number_marker += 1
                    else:
                        self.bot_marker_publisher_3.publish(marker_link)
                        bot_number_marker += 1
            elif bot_number_marker < 15:
                self.bot_marker_publisher_3.publish(marker_cap_1)
                bot_number_marker += 1
                if p1.tolist() != p2.tolist():
                    if bot_number_marker < 15:
                        self.bot_marker_publisher_3.publish(marker_cap_2)
                        bot_number_marker += 1
                    else:
                        self.bot_marker_publisher_4.publish(marker_cap_2)
                        bot_number_marker += 1
                    if bot_number_marker < 15:
                        self.bot_marker_publisher_3.publish(marker_link)
                        bot_number_marker += 1
                    else:
                        self.bot_marker_publisher_4.publish(marker_link)
                        bot_number_marker += 1
            elif bot_number_marker < 20:
                self.bot_marker_publisher_4.publish(marker_cap_1)
                bot_number_marker += 1
                if p1.tolist() != p2.tolist():
                    if bot_number_marker < 20:
                        self.bot_marker_publisher_4.publish(marker_cap_2)
                        bot_number_marker += 1
                    else:
                        self.bot_marker_publisher_5.publish(marker_cap_2)
                        bot_number_marker += 1
                    if bot_number_marker < 20:
                        self.bot_marker_publisher_4.publish(marker_link)
                        bot_number_marker += 1
                    else:
                        self.bot_marker_publisher_5.publish(marker_link)
                        bot_number_marker += 1
            else:
                self.bot_marker_publisher_5.publish(marker_cap_1)
                if p1.tolist() != p2.tolist():
                    self.bot_marker_publisher_5.publish(marker_cap_2)
                    self.bot_marker_publisher_5.publish(marker_link)

        # Publish end-effector trajectory as line strip
        if self.ee_position_history:
            traj_marker = Marker()
            traj_marker.header.frame_id = "world"
            traj_marker.header.stamp = self.get_clock().now().to_msg()
            traj_marker.ns = "ee_trajectory"
            traj_marker.id = 0
            traj_marker.type = Marker.LINE_STRIP
            traj_marker.action = Marker.ADD

            traj_marker.points = self.ee_position_history

            traj_marker.scale.x = 0.01
            traj_marker.color.r = 0.0
            traj_marker.color.g = 1.0
            traj_marker.color.b = 0.8
            traj_marker.color.a = 1.0
            
            traj_marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg() 

            self.trajectory_marker_publisher.publish(traj_marker)

    # =========================================================================
    # DATA LOGGING AND PLOTTING
    # =========================================================================

    def save_and_plot_results(self):
        """
        Save logged data to CSV and generate plots for analysis.
        Called at end of simulation.
        """
        if not self.history_data_frames:
            self.get_logger().warn("No data to save or plot.")
            return
        
        # Create output directory
        output_pack = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "sim")
        os.makedirs(output_pack, exist_ok=True)
        output_dir = os.path.join(output_pack, self.output_basename)
        os.makedirs(output_dir, exist_ok=True)

        # Save raw data to CSV
        df = pd.DataFrame(self.history_data_frames)
        df['final_run_status'] = self.final_run_status

        csv_filename = os.path.join(output_dir, "run_data.csv")
        df.to_csv(csv_filename, index=False)
        
        # Extract time series data for plotting
        time_history = [d['time'] for d in self.history_data_frames]
        h_min_history = [d['min_h'] for d in self.history_data_frames]
        psi_min_history = [d['min_psi'] for d in self.history_data_frames]
        qp_infeasible_history = [d['qp_infeasible'] for d in self.history_data_frames]
        
        # Per-joint data
        joint_q_history_extracted = {name: [] for name in self.robot_joint_names}
        next_q_history_extracted = {name: [] for name in self.robot_joint_names}
        joint_dq_history_extracted = {name: [] for name in self.robot_joint_names}
        next_dq_history_extracted = {name: [] for name in self.robot_joint_names}
        joint_ddq_history_extracted = {name: [] for name in self.robot_joint_names}
        joint_ddq_nominal_history_extracted = {name: [] for name in self.robot_joint_names}

        # Dynamic parameter history
        current_gamma_js_history = [d['current_gamma_js'] for d in self.history_data_frames]
        max_gamma_required_for_h_history = [d['max_gamma_required_for_h'] for d in self.history_data_frames]
        current_beta_js_history = [d['current_beta_js'] for d in self.history_data_frames]
        max_beta_required_for_psi_history = [d['max_beta_required_for_psi'] for d in self.history_data_frames]

        solve_time_history = [d['solve_time'] for d in self.history_data_frames]
        
        # Individual h and psi values per constraint
        all_h_kj_plots = {} 
        all_psi_kj_plots = {}

        if self.history_data_frames:
            sample_data = self.history_data_frames[0]
            for key in sample_data.keys():
                if key.startswith('h_') and key not in ['h_min']:
                    all_h_kj_plots[key] = []
                elif key.startswith('psi_') and key not in ['psi_min']:
                    all_psi_kj_plots[key] = []

        # Extract all time series
        for data_point in self.history_data_frames:
            for i, joint_name in enumerate(self.robot_joint_names):
                joint_q_history_extracted[joint_name].append(data_point['joint_q'][i])
                next_q_history_extracted[joint_name].append(data_point['next_q'][i])
                joint_dq_history_extracted[joint_name].append(data_point['joint_dq'][i])
                next_dq_history_extracted[joint_name].append(data_point['next_dq'][i])
                joint_ddq_history_extracted[joint_name].append(data_point['joint_ddq'][i])
                joint_ddq_nominal_history_extracted[joint_name].append(data_point['joint_ddq_nominal'][i])
            
            for key in all_h_kj_plots.keys():
                all_h_kj_plots[key].append(data_point[key])
            for key in all_psi_kj_plots.keys():
                all_psi_kj_plots[key].append(data_point[key])

        # Plot 1: Overall CBF values
        fig_overview = plt.figure(figsize=(10, 6))
        ax_overview = fig_overview.add_subplot(111) 
        
        ax_overview.plot(time_history, h_min_history, label='Min $h_{kj}$')
        ax_overview.plot(time_history, psi_min_history, label='Min $\\psi_{kj}$')
        
        infeasible_times = [time_history[i] for i, is_infeasible in enumerate(qp_infeasible_history) if is_infeasible]
        for t_inf in infeasible_times:
            ax_overview.axvline(t_inf, color='magenta', linestyle=':',alpha=0.1, lw=0.8, label='QP Infeasible' if t_inf == infeasible_times[0] else "") 

        ax_overview.axhline(0, color='r', linestyle='--', label='Safety Boundary (0)')
        ax_overview.set_xlabel('Time (s)')
        ax_overview.set_ylabel('CBF Value')
        ax_overview.set_title('Overall Minimum CBF Values Over Time')
        ax_overview.legend()
        ax_overview.grid(True)
        fig_overview.tight_layout()

        base_filename = os.path.join(output_dir, "plot")
        fig_overview.savefig(f"{base_filename}_cbf_overview.png")

        # Plot 2: Joint states (position, velocity, acceleration)
        num_joints = len(self.robot_joint_names)
        fig_joint_data, axs_joint_data = plt.subplots(num_joints, 3, figsize=(15, 3 * num_joints), sharex=True)
        fig_joint_data.suptitle('Joint States and Accelerations Over Time', fontsize=16)

        for i, joint_name in enumerate(self.robot_joint_names):
            # Position
            axs_joint_data[i, 0].plot(time_history, joint_q_history_extracted[joint_name], label=f'{joint_name} ($q$)')
            axs_joint_data[i, 0].plot(time_history, next_q_history_extracted[joint_name], label=f'{joint_name} (Next $q$)', alpha=0.5)
            axs_joint_data[i, 0].axhline(q_max_arm[i], color='k', linestyle=':', alpha=0.7, label='Max Position' if i==0 else "")
            axs_joint_data[i, 0].axhline(q_min_arm[i], color='k', linestyle=':', alpha=0.7, label='Min Position' if i==0 else "")
            axs_joint_data[i, 0].set_ylabel('Position (rad)')
            axs_joint_data[i, 0].grid(True)
            if i == 0: axs_joint_data[i, 0].legend()

            # Velocity
            axs_joint_data[i, 1].plot(time_history, joint_dq_history_extracted[joint_name], label=f'{joint_name} ($\\dot{{q}}$)')
            axs_joint_data[i, 1].plot(time_history, next_dq_history_extracted[joint_name], label=f'{joint_name} (Next $\\dot{{q}}$)', alpha=0.5)
            axs_joint_data[i, 1].axhline(dq_max_arm[i], color='k', linestyle=':', alpha=0.7, label='Max Vel' if i==0 else "")
            axs_joint_data[i, 1].axhline(dq_min_arm[i], color='k', linestyle=':', alpha=0.7, label='Min Vel' if i==0 else "")
            axs_joint_data[i, 1].set_ylabel('Velocity (rad/s)')
            axs_joint_data[i, 1].grid(True)
            if i == 0: axs_joint_data[i, 1].legend()

            # Acceleration
            axs_joint_data[i, 2].plot(time_history, joint_ddq_history_extracted[joint_name], label=f'{joint_name} ($\\ddot{{q}}$ Applied)')
            axs_joint_data[i, 2].plot(time_history, joint_ddq_nominal_history_extracted[joint_name], label=f'{joint_name} ($\\ddot{{q}}$ Nominal)', alpha=0.7)
            axs_joint_data[i, 2].axhline(ddq_max_arm[i], color='k', linestyle=':', alpha=0.7, label='Max Accel' if i==0 else "")
            axs_joint_data[i, 2].axhline(ddq_min_arm[i], color='k', linestyle=':', alpha=0.7, label='Min Accel' if i==0 else "")
            axs_joint_data[i, 2].set_ylabel('Acceleration (rad/s$^2$)')
            axs_joint_data[i, 2].set_ylim(ddq_min_arm[i]*1.1, ddq_max_arm[i]*1.1)
            axs_joint_data[i, 2].grid(True)
            if i == 0: axs_joint_data[i, 2].legend()
            
            for t_inf in infeasible_times:
                axs_joint_data[i, 2].axvline(t_inf, color='magenta', linestyle=':', lw=0.6, alpha=0.1, label='QP Infeasible' if t_inf == infeasible_times[0] and i==0 else "")

        axs_joint_data[-1, 0].set_xlabel('Time (s)')
        axs_joint_data[-1, 1].set_xlabel('Time (s)')
        axs_joint_data[-1, 2].set_xlabel('Time (s)')
        
        fig_joint_data.tight_layout(rect=[0, 0, 1, 0.96])
        fig_joint_data.savefig(f"{base_filename}_joint_states.png")

        # Plot 3: Joint accelerations (larger view)
        fig_accel, axs_accel = plt.subplots(num_joints, 1, figsize=(10, 14), sharex=True)
        fig_accel.suptitle('Joint Accelerations Over Time', fontsize=16)

        for i, joint_name in enumerate(self.robot_joint_names):
            axs_accel[i].plot(time_history, joint_ddq_history_extracted[joint_name], label=f'{joint_name} (Applied)')
            axs_accel[i].plot(time_history, joint_ddq_nominal_history_extracted[joint_name], label='Nominal', alpha=0.7)
            axs_accel[i].axhline(ddq_max_arm[i], color='r', linestyle=':', alpha=0.5)
            axs_accel[i].axhline(ddq_min_arm[i], color='r', linestyle=':', alpha=0.5)
            
            for t_inf in infeasible_times:
                axs_accel[i].axvline(t_inf, color='magenta', linestyle=':', lw=0.8, alpha=0.2)
                
            axs_accel[i].set_ylabel('Accel (rad/s²)')
            axs_accel[i].legend(loc='upper right')
            axs_accel[i].grid(True)
            axs_accel[i].set_ylim(ddq_min_arm[i] * 1.5, ddq_max_arm[i] * 1.5)

        if infeasible_times:
            axs_accel[0].axvline(infeasible_times[0], color='magenta', linestyle=':', lw=0.8, alpha=0.2, label='QP Infeasible')
            axs_accel[0].legend(loc='upper right')

        axs_accel[-1].set_xlabel('Time (s)')
        fig_accel.tight_layout(rect=[0, 0.03, 1, 0.96])
        fig_accel.savefig(f"{base_filename}_joint_accelerations.png")

        # Plot 4: All individual h values
        fig_all_h, ax_all_h = plt.subplots(figsize=(12, 8))
        ax_all_h.set_title('All $h_{kj}$ Values Over Time (Control Point-Obstacle Pairs)')
        ax_all_h.set_xlabel('Time (s)')
        ax_all_h.set_ylabel('$h_{kj}$ Value')
        ax_all_h.axhline(0, color='r', linestyle='--', label='Safety Boundary (0)')
        ax_all_h.grid(True)

        colors = plt.cm.get_cmap('tab10', len(all_h_kj_plots))
        for i, (key, h_values) in enumerate(all_h_kj_plots.items()):
            ax_all_h.plot(time_history, h_values, label=key, color=colors(i))
        ax_all_h.legend(loc='best', ncol=2, fontsize='small')
        ax_all_h.set_ylim(-0.3, 1)
        fig_all_h.tight_layout()
        fig_all_h.savefig(f"{base_filename}_all_h_values.png")

        # Plot 5: All individual psi values
        fig_all_psi, ax_all_psi = plt.subplots(figsize=(12, 8))
        ax_all_psi.set_title('All $\\psi_{kj}$ Values Over Time (Control Point-Obstacle Pairs)')
        ax_all_psi.set_xlabel('Time (s)')
        ax_all_psi.set_ylabel('$\\psi_{kj}$ Value')
        ax_all_psi.axhline(0, color='r', linestyle='--', label='Safety Boundary (0)')
        ax_all_psi.grid(True)

        colors = plt.cm.get_cmap('tab10', len(all_psi_kj_plots))
        for i, (key, psi_values) in enumerate(all_psi_kj_plots.items()):
            ax_all_psi.plot(time_history, psi_values, label=key, color=colors(i))
        ax_all_psi.legend(loc='best', ncol=2, fontsize='small')
        ax_all_psi.set_ylim(-0.3, 1)
        fig_all_psi.tight_layout()
        fig_all_psi.savefig(f"{base_filename}_all_psi_values.png")

        # Plot 6: Dynamic gamma parameter
        fig_gamma_dynamic, ax_gamma_dynamic = plt.subplots(figsize=(10, 6))
        ax_gamma_dynamic.plot(time_history, current_gamma_js_history, label='Applied $\\gamma_{js}$')
        ax_gamma_dynamic.plot(time_history, max_gamma_required_for_h_history, '--', label='Max Required $\\gamma_{js}$')
        ax_gamma_dynamic.set_xlabel('Time (s)')
        ax_gamma_dynamic.set_ylabel('$\\gamma_{js}$ Value')
        ax_gamma_dynamic.set_title('Dynamic $\\gamma_{js}$ Over Time')
        ax_gamma_dynamic.legend()
        ax_gamma_dynamic.grid(True)
        ax_gamma_dynamic.set_ylim(0, GAMMA_MAX_LIMIT * 1.1)
        fig_gamma_dynamic.tight_layout()
        fig_gamma_dynamic.savefig(f"{base_filename}_dynamic_gamma.png")

        # Plot 7: Dynamic beta parameter
        fig_beta_dynamic, ax_beta_dynamic = plt.subplots(figsize=(10, 6))
        ax_beta_dynamic.plot(time_history, current_beta_js_history, label='Applied $\\\\beta_{js}$')
        ax_beta_dynamic.plot(time_history, max_beta_required_for_psi_history, '--', label='Max Required $\\\\beta_{js}$')
        ax_beta_dynamic.set_xlabel('Time (s)')
        ax_beta_dynamic.set_ylabel('$\\\\beta_{js}$ Value')
        ax_beta_dynamic.set_title('Dynamic $\\\\beta_{js}$ Over Time')
        ax_beta_dynamic.legend()
        ax_beta_dynamic.grid(True)
        ax_beta_dynamic.set_ylim(0, BETA_MAX_LIMIT * 1.1)
        fig_beta_dynamic.tight_layout()
        fig_beta_dynamic.savefig(f"{base_filename}_dynamic_beta.png")

        # Plot 8: QP solve times
        fig_sol_times, ax_sol_times = plt.subplots(figsize=(10, 6))
        ax_sol_times.plot(time_history, solve_time_history, label='Solve time (s)')
        ax_sol_times.set_xlabel('Time (s)')
        ax_sol_times.set_ylabel('Solve Time (s)')
        ax_sol_times.set_title('Solve Time Over Time')
        ax_sol_times.legend()
        ax_sol_times.grid(True)
        fig_sol_times.tight_layout()
        fig_sol_times.savefig(f"{base_filename}_solve_time.png")

        self.get_logger().info(f"All plots saved to {output_dir}")
        plt.close('all')

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args=None):
    """Main function to run the CBF safety filter node."""
    rclpy.init(args=args)
    node = CbfControllerNode()
    try:
        # Spin node until shutdown is initiated
        while rclpy.ok() and not node.is_shutdown_initiated:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # Save results on shutdown
        node.save_and_plot_results()
        if not node.is_shutdown_initiated:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()