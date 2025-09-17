#!/usr/bin/env python3

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

# --- Pinocchio Model Setup ---
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

EE_FRAME_NAME = "fr3_hand_tcp"
try:
    EE_FRAME_ID = model.getFrameId(EE_FRAME_NAME)
except IndexError:
    rclpy.logging.get_logger('cbf_controller_node').error(f"Error: End-effector frame '{EE_FRAME_NAME}' not found in URDF.")
    rclpy.logging.get_logger('cbf_controller_node').error(f"Available frames: {[f.name for f in model.frames]}")
    exit()


# --- Define Control links on the Robot ---
links_def = [
    # {'name': 'link2_base', 'start_frame_name': 'fr3_link2_offset1', 'end_frame_name': 'fr3_link2_offset2', 'radius': 0.055}, # Uncomment for more accurate collision checking
    {'name': 'link2', 'start_frame_name': 'fr3_link2', 'end_frame_name': 'fr3_link3', 'radius': 0.06},
    {'name': 'joint4',   'start_frame_name': 'fr3_link4', 'end_frame_name': 'fr3_link5_offset1', 'radius': 0.065},
    {'name': 'forearm1',   'start_frame_name': 'fr3_link5_offset2', 'end_frame_name': 'fr3_link5_offset3', 'radius': 0.035},
    {'name': 'forearm2',   'start_frame_name': 'fr3_link5_offset3', 'end_frame_name': 'fr3_link5', 'radius': 0.05},
    {'name': 'wrist',     'start_frame_name': 'fr3_link7', 'end_frame_name': 'fr3_hand',  'radius': 0.055}, # Change fr3_linkt to fr3_link7_offset1 for more accurate collision checking
    {'name': 'hand',     'start_frame_name': 'fr3_hand_offset1', 'end_frame_name': 'fr3_hand_offset2',  'radius': 0.03},
    {'name': 'end_effector',      'start_frame_name': EE_FRAME_NAME,  'end_frame_name': EE_FRAME_NAME, 'radius': 0.03},
]
active_links = []

self_collision_link_pair = [
    ('link2', 'hand'),
    ('link2', 'end_effector'),
    ('link2', 'forearm1')
]

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

ddq_max_scalar = 40.0  # rad/s^2 (Adjusted from 10.0*0.5 to a more reasonable value for simulation)
ddq_max_arm = np.full(NUM_ARM_JOINTS, ddq_max_scalar)
ddq_min_arm = np.full(NUM_ARM_JOINTS, -ddq_max_scalar)

# Joint velocity limits (from your script)
dq_max_arm = np.array([2.0, 1.0, 1.5, 1.25, 3.0, 1.5, 3.0]) # rad/s
dq_min_arm = - dq_max_arm # Symmetric limits

q_max_arm = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
q_min_arm = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])

ACCEL_DIF_THRESHOLD = 10  # rad/s^2. If total |ddq| is less than this, try to recover.

# --- Constants for dynamic gamma/beta calculation ---
EPSILON_DENOMINATOR = 1e-6 # Avoid division by zero when h or psi are very close to zero
GAMMA_ADJUST_BUFFER = 0.5 # Small buffer to add to min_gamma_required to prevent numerical issues or oscillations
GAMMA_MAX_LIMIT = 200.0 # Upper limit for dynamically adjusted gamma_js (prevent it from exploding)
BETA_ADJUST_BUFFER = 0.5 # Small buffer to add to min_beta_required
BETA_MAX_LIMIT = 250.0 # Upper limit for dynamically adjusted beta_js

# --- Geometry Helper for Capsule-to-Capsule distance ---
def get_closest_points_between_segments(p1, p2, q1, q2):
    """
    Calculates the closest points between two line segments (p1, p2) and (q1, q2).
    This is a robust implementation that correctly handles all edge cases.

    Args:
        p1, p2 (np.array): Endpoints of the first segment.
        q1, q2 (np.array): Endpoints of the second segment.

    Returns:
        tuple: (c1, c2, s) where c1 is the closest point on segment 1, 
               c2 is the closest point on segment 2, and s is the interpolation
               factor for the first segment (the robot link).
    """
    u = p2 - p1
    v = q2 - q1
    w = p1 - q1

    a = np.dot(u, u)  # |u|^2
    b = np.dot(u, v)
    c = np.dot(v, v)  # |v|^2
    d = np.dot(u, w)
    e = np.dot(v, w)

    D = a * c - b * b  # D = |u|^2 * |v|^2 - |u.v|^2 = |u|^2 * |v|^2 * sin^2(theta)
    
    s_c, t_c = D, D  # Numerators for s and t
    s, t = 0.0, 0.0    # Final parameters

    # Compute the line parameters of the two closest points on the infinite lines
    # connecting vector of the closest points: dist = (p1 + s * u) - (q1 + t * v)
    # Both u and v must be parallel to the connecting vector dist * u = 0 and dist * v = 0
    # Which yelds the equations:
    # s = (b*e - c*d) / D
    # t = (a*e - b*d) / D
    if D < 1e-7:  # Lines are parallel
        s = np.clip(-d / a, 0.0, 1.0) if a > 1e-7 else 0.0
        t = (b * s + e) / c if c > 1e-7 else 0.0
    else:
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
    
    return c1, c2, s


# --- Joint Space HOCBF Functions ---
def h_func(p_rel, robot_link_radius, obstacle_radius, d_margin=0):
    """Barrier function h(x) for two capsules."""
    R_eff_sq = (robot_link_radius + obstacle_radius + d_margin)**2
    return np.dot(p_rel, p_rel) - R_eff_sq

def Lf_h(p_rel, v_rel):
    """Lie derivative of h along f (drift dynamics). v_rel = v_robot - v_obstacle."""
    return 2 * np.dot(p_rel, v_rel)

def psi_func(h_val, Lf_h_val, gamma_param):
    return Lf_h_val + gamma_param * h_val

def Lf_psi(v_rel, a_drift_rel, p_rel, Lf_h_val, gamma_param):
    """Lie derivative of psi along f. a_drift_rel = a_robot - a_obstacle."""
    term_vel_sq = 2 * np.dot(v_rel, v_rel)
    term_accel = 2 * np.dot(p_rel, a_drift_rel)
    return term_vel_sq + term_accel + gamma_param * Lf_h_val

def Lg_psi(J_p_robot_closest, p_rel):
    """Lie derivative of psi along g (actuation)."""
    J_p_arm = J_p_robot_closest[:, :NUM_ARM_JOINTS]
    return 2 * np.dot(p_rel, J_p_arm)


# Helper to avoid code duplication between QP solver and dynamic param calculation
def get_point_kinematics(start_frame_id, end_frame_id, t, dq_full_curr):
    J_full_p1 = pin.getFrameJacobian(model, data, start_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
    J_full_p2 = pin.getFrameJacobian(model, data, end_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
    J_C = (1 - t) * J_full_p1 + t * J_full_p2

    a_drift_p1 = pin.getFrameClassicalAcceleration(model, data, start_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
    a_drift_p2 = pin.getFrameClassicalAcceleration(model, data, end_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
    a_drift_C = (1 - t) * a_drift_p1 + t * a_drift_p2

    v_C = J_C @ dq_full_curr
    return J_C, v_C, a_drift_C

# --- Nominal Controller (Joint Space PD to Cartesian Goal) - Copied from your CBF script ---
def nominal_controller_js(q_arm_curr, dq_arm_curr, target_ee_pos_cartesian, target):
    Kp_cart = 500.0 # Your current Kp
    Kd_cart = 50.0 # Your current Kd

    q_full_curr = np.zeros(model.nq)
    q_full_curr[:NUM_ARM_JOINTS] = q_arm_curr
    dq_full_curr = np.zeros(model.nv)
    dq_full_curr[:NUM_ARM_JOINTS] = dq_arm_curr

    # Perform kinematics once for nominal controller
    pin.forwardKinematics(model, data, q_full_curr, dq_full_curr, np.zeros(model.nv))
    pin.computeJointJacobians(model, data, q_full_curr)
    pin.updateFramePlacements(model, data)
    
    current_ee_pos = data.oMf[EE_FRAME_ID].translation
    J_ee_full = pin.getFrameJacobian(model, data, EE_FRAME_ID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J_ee_p_arm = J_ee_full[:3, :NUM_ARM_JOINTS]

    alpha = 0.01
    target = target_ee_pos_cartesian
    target = (1-alpha)*target + alpha*target_ee_pos_cartesian
    error_pos_cart = target - current_ee_pos
    current_ee_vel_cart = J_ee_p_arm @ dq_arm_curr
    error_vel_cart = -current_ee_vel_cart

    f_desired_cart = Kp_cart * error_pos_cart + Kd_cart * error_vel_cart

    lambda_damp = 0.01 # Damping for pseudo-inverse
    # Use np.linalg.pinv directly as it's more robust
    J_pseudo_inv = np.linalg.pinv(J_ee_p_arm, rcond=lambda_damp)
    
    #ddq_nominal_arm = J_ee_p_arm.T @ f_desired_cart 
    ddq_primary_task = J_pseudo_inv @ f_desired_cart

    # Joint Limit Avoidance
    Kp_joint_limit = 200.0  # Gain for the repulsive force
    activation_threshold = 0.9 # (e.g., 0.9 means start avoiding at 90% of the joint range)

    # Calculate the middle point and range for each joint
    q_mid = (q_max_arm + q_min_arm) / 2.0
    q_range = q_max_arm - q_min_arm
    
    ddq_secondary_task = np.zeros(NUM_ARM_JOINTS)
    
    for i in range(NUM_ARM_JOINTS):
        # Check if the joint is within the activation zone near its limits
        if abs(q_arm_curr[i] - q_mid[i]) / q_range[i] > (activation_threshold / 2.0):
            # This calculates a "repulsive" acceleration that pushes the joint back towards its center
            gradient = -2 * (q_arm_curr[i] - q_mid[i]) / (q_range[i]**2)
            ddq_secondary_task[i] = Kp_joint_limit * gradient

    # --- Null-Space Projection ---

    # 1. Calculate the null-space projector for the Jacobian
    I = np.identity(NUM_ARM_JOINTS)
    null_space_projector = I - J_pseudo_inv @ J_ee_p_arm
    
    # 2. Project the secondary task into the null space
    ddq_secondary_projected = null_space_projector @ ddq_secondary_task
        
    ddq_nominal_arm = ddq_primary_task + ddq_secondary_projected
    
    # ddq_nominal_arm = np.clip(ddq_nominal_arm, ddq_min_arm, ddq_max_arm)
    return ddq_nominal_arm, target


# --- Nominal Controller (MPC to Cartesian Goal) ---
def nominal_controller_mpc(q_arm_curr, dq_arm_curr, target_ee_pos_cartesian, dt):
    """
    A simple kinematic MPC using CVXPY to generate a nominal joint acceleration command.
    This controller respects joint limits but is blind to obstacles.
    """
    # --- MPC Parameters (Tuned for stability and speed) ---
    N = 50          # Prediction horizon
    Q_pos = 10.0  # Weight for end-effector position error
    R_accel = 0.0001   # Weight for control effort
    W_stop = 0.001   # Weight for stopping cost
    W_jerk = 0.001    # Weight for jerk cost

    # --- Setup the Optimization Problem with CVXPY ---
    # Decision variables (control inputs)
    ddq = cp.Variable((NUM_ARM_JOINTS, N))

    # State variables (predicted trajectory)
    q = cp.Variable((NUM_ARM_JOINTS, N + 1))
    dq = cp.Variable((NUM_ARM_JOINTS, N + 1))

    # --- Build Cost and Constraints ---
    cost = 0
    constraints = []

    # Penalize large control inputs
    cost += R_accel * cp.sum_squares(ddq)

    # Initial state constraint
    constraints += [q[:, 0] == q_arm_curr]
    constraints += [dq[:, 0] == dq_arm_curr]

    # System dynamics, state/input constraints, and costs over the horizon
    max_decel = np.abs(ddq_min_arm)
    for k in range(N):
        # Kinematic integration (Euler)
        q_next = q[:, k] + dq[:, k] * dt + 0.5 * ddq[:, k] * dt**2
        dq_next = dq[:, k] + ddq[:, k] * dt

        constraints += [q[:, k + 1] == q_next]
        constraints += [dq[:, k + 1] == dq_next]

        # Enforce all joint limits on the predicted trajectory
        constraints += [q_min_arm <= q[:, k + 1], q[:, k + 1] <= q_max_arm]
        constraints += [dq_min_arm <= dq[:, k + 1], dq[:, k + 1] <= dq_max_arm]
        constraints += [ddq_min_arm <= ddq[:, k], ddq[:, k] <= ddq_max_arm]

        # Stop Cost
        time_to_stop = (N - 1 - k) * dt
        dq_max_stoppable = max_decel * time_to_stop
        velocity_overshoot = cp.pos(cp.abs(dq[:, k+1]) - dq_max_stoppable)
        # cost += W_stop * cp.sum_squares(velocity_overshoot)

        # Jerk Cost
        # if k > 0:
        #    cost += W_jerk * cp.sum_squares(ddq[:, k] - ddq[:, k-1])

    # --- Terminal Cost (End-Effector Position) ---
    q_full_pin = np.zeros(model.nq)
    q_full_pin[:NUM_ARM_JOINTS] = q_arm_curr
    pin.forwardKinematics(model, data, q_full_pin)
    pin.computeJointJacobians(model, data, q_full_pin)
    pin.updateFramePlacements(model, data)
    
    J_ee_full = pin.getFrameJacobian(model, data, EE_FRAME_ID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J_ee_p_arm = J_ee_full[:3, :NUM_ARM_JOINTS]
    current_ee_pos = data.oMf[EE_FRAME_ID].translation

    # Linearized prediction of final EE position
    predicted_ee_pos = current_ee_pos + J_ee_p_arm @ (q[:, N] - q_arm_curr)
    error_pos_cart = predicted_ee_pos - target_ee_pos_cartesian
    cost += Q_pos * cp.sum_squares(error_pos_cart)

    # Terminal Constraint
    constraints += [cp.norm(error_pos_cart, 2) <= 0.05]  # 5 cm tolerance at horizon end
    # constraints += [cp.norm(dq[:, N], 'inf') <= 0.05]  # Also try to be nearly stopped at horizon end

    # --- Solve the QP ---
    try:
        problem = cp.Problem(cp.Minimize(cost), constraints)
        # Use OSQP, which is already working in your CBF filter
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        
        if ddq.value is not None:
             # Return only the FIRST acceleration command from the optimal sequence
            return ddq.value[:, 0]
        else:
            # Solver finished but found no solution (infeasible)
            print("MPC problem is infeasible. Returning zero acceleration.")
            return np.zeros(NUM_ARM_JOINTS)

    except Exception as e:
        # Solver failed catastrophically
        print(f"MPC Solver failed with error: {e}. Returning zero acceleration.")
        return np.zeros(NUM_ARM_JOINTS)
'''
    
import casadi as ca
def nominal_controller_mpc(q_arm_curr, dq_arm_curr, target_ee_pos_cartesian, dt):
    """
    A simple kinematic MPC to generate a nominal joint acceleration command.
    This controller respects joint limits but is blind to obstacles.
    """
    # --- MPC Parameters (Tuned for stability and speed) ---
    N = 15          # Prediction horizon (longer horizon for smoother plans)
    Q_pos = 100.0  # Weight for end-effector position error (reduced for less aggression)
    R_accel = 0.001   # Weight for control effort (encourages smaller accelerations)
    W_stop = 0.01   # Weight for stopping cost (dampens motion near goal)
    W_jerk = 0.05    # Weight for jerk cost (encourages smooth accelerations)

    # --- Setup the Optimization Problem with CasADi ---
    opti = ca.Opti()

    # Decision variables (control inputs)
    ddq = opti.variable(NUM_ARM_JOINTS, N)

    # State variables (predicted trajectory)
    q = opti.variable(NUM_ARM_JOINTS, N + 1)
    dq = opti.variable(NUM_ARM_JOINTS, N + 1)

    # --- Cost Function ---
    cost = 0
    # cost += R_accel * ca.sumsqr(ddq) # Penalize large control inputs

    # --- Constraints ---
    # Initial state constraint
    opti.subject_to(q[:, 0] == q_arm_curr)
    opti.subject_to(dq[:, 0] == dq_arm_curr)

    max_decel = np.abs(ddq_min_arm)
    stop_cost = 0
    jerk_cost = 0

    # System dynamics and state/input constraints over the horizon
    for k in range(N):
        # Kinematic integration (Euler)
        q_next = q[:, k] + dq[:, k] * dt + 0.5 * ddq[:, k] * dt**2
        dq_next = dq[:, k] + ddq[:, k] * dt

        opti.subject_to(q[:, k + 1] == q_next)
        opti.subject_to(dq[:, k + 1] == dq_next)

        # CRITICAL: Enforce all joint limits on the predicted trajectory
        opti.subject_to(opti.bounded(q_min_arm, q[:, k + 1], q_max_arm))
        opti.subject_to(opti.bounded(dq_min_arm, dq[:, k + 1], dq_max_arm))
        opti.subject_to(opti.bounded(ddq_min_arm, ddq[:, k], ddq_max_arm)) # Re-enable this!

        # Stop Cost
        time_to_stop = (N - 1 - k) * dt
        dq_max_stoppable = max_decel * time_to_stop
        velocity_overshoot = ca.fmax(0, ca.fabs(dq[:, k+1]) - dq_max_stoppable)
        stop_cost += ca.sumsqr(velocity_overshoot)

        # Jerk Cost (applied to the decision variables)
        if k > 0:
            jerk_cost += ca.sumsqr(ddq[:, k] - ddq[:, k-1])

    # cost += W_stop * stop_cost
    # cost += W_jerk * jerk_cost

        # --- Terminal Cost (End-Effector Position) ---
        q_full_pin = np.zeros(model.nq)
        q_full_pin[:NUM_ARM_JOINTS] = q_arm_curr
        pin.forwardKinematics(model, data, q_full_pin)
        pin.computeJointJacobians(model, data, q_full_pin)
        pin.updateFramePlacements(model, data)
        
        J_ee_full = pin.getFrameJacobian(model, data, EE_FRAME_ID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_ee_p_arm = J_ee_full[:3, :NUM_ARM_JOINTS]
        current_ee_pos = data.oMf[EE_FRAME_ID].translation

        predicted_ee_pos = current_ee_pos + J_ee_p_arm @ (q[:, N] - q_arm_curr)
        error_pos_cart = predicted_ee_pos - target_ee_pos_cartesian
        cost += Q_pos*k * ca.sumsqr(error_pos_cart)

    # Terminal Constraint
    opti.subject_to(error_pos_cart < 0.01)

    opti.minimize(cost)

    # --- Solve the QP ---
    p_opts = {"expand": True}
    # Use 'qpoases' for a massive speedup on this QP problem
    s_opts = {"print_level": 0, "sb": "yes"} 
    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol = opti.solve()
        ddq_nominal_arm = sol.value(ddq[:, 0])
        # Double integration to get joint position
        q_nominal_arm = q_arm_curr + dq_arm_curr*dt + 0.5*ddq_nominal_arm**2
        return ddq_nominal_arm
    except RuntimeError:
        print("MPC Solver failed. Returning zero acceleration.")
        return np.zeros(NUM_ARM_JOINTS)
'''

# --- HOCBF-QP Safety Filter ---
def solve_hocbf_qp_js(dt_val, q_full_curr, dq_full_curr, ddq_nominal_arm_val, current_gamma_js_val, current_beta_js_val, d_margin, current_obstacles_list_sim, active_links_list, node_logger):
    u_qp_js = cp.Variable(NUM_ARM_JOINTS)

    ddq_nominal_arm_np = np.array(ddq_nominal_arm_val).flatten()

    cost = cp.sum_squares(u_qp_js - ddq_nominal_arm_np)
    constraints_qp = []
    qp_failed_flag = False

    # Pre-calculate all necessary kinematics once per control loop for all CPs
    # This prevents redundant Pinocchio computations within the nested loops.
    ddq_full_zeros = np.zeros(model.nv)
    pin.forwardKinematics(model, data, q_full_curr, dq_full_curr, ddq_full_zeros)
    pin.computeJointJacobians(model, data, q_full_curr)
    pin.updateFramePlacements(model, data)

    # Iterate through each defined LINK and each obstacle
    for link_info in active_links_list:
        start_frame_id = link_info['start_frame_id']
        end_frame_id = link_info['end_frame_id']
        link_radius_val = link_info['radius']

        # Get 3D positions of the link's start and end points
        p1 = data.oMf[start_frame_id].translation
        p2 = data.oMf[end_frame_id].translation

        for obs_item in current_obstacles_list_sim:
            # --- 1. FFind the closest points between the two capsule segments ---
            c_robot, c_obs, t = get_closest_points_between_segments(
                p1, p2, obs_item['pose_start'], obs_item['pose_end']
            )
            
            p_rel = c_robot - c_obs
            h_val = h_func(p_rel, link_radius_val, obs_item['radius'], d_margin)
            if h_val < 0.07: # Only add constraints if h is small (indicating a risk of collision) [0.15m^2 * 3 = 0.0675]

                # --- 2. Calculate kinematics for the closest point on the ROBOT ---
                J_C, v_C, a_drift_C = get_point_kinematics(
                    link_info['start_frame_id'], link_info['end_frame_id'], t, dq_full_curr
                )

                # --- 3. Calculate relative velocity and acceleration drift ---
                # Assuming obstacle velocity is constant, so acceleration is zero
                v_rel = v_C - obs_item['velocity']
                a_drift_rel = a_drift_C # - a_obs_drift (which is 0)

                # --- 4. Use the new HOCBF functions for capsules ---
                Lf_h_val = Lf_h(p_rel, v_rel)
                psi_val = psi_func(h_val, Lf_h_val, current_gamma_js_val)
                Lf_psi_val = Lf_psi(v_rel, a_drift_rel, p_rel, Lf_h_val, current_gamma_js_val)
                Lg_psi_val_arm = Lg_psi(J_C, p_rel)

                constraints_qp.append(Lg_psi_val_arm @ u_qp_js >= -Lf_psi_val - current_beta_js_val * psi_val)

    # Iterate through self-collision pairs
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

        c_link1, c_link2, t = get_closest_points_between_segments(p1_1, p2_1, p1_2, p2_2)
        p_rel = c_link1 - c_link2
        h_val = h_func(p_rel, link_radius_val_1, link_radius_val_2)
        if h_val < 0.03:  # Only add constraints if h is small (indicating a risk of self-collision)
            # Calculate kinematics for the closest point on the first link
            J_C1, v_C1, a_drift_C1 = get_point_kinematics(
                start_frame_id_1, end_frame_id_1, t, dq_full_curr
            )
            J_C2, v_C2, a_drift_C2 = get_point_kinematics(
                start_frame_id_2, end_frame_id_2, t, dq_full_curr
            )
            # Calculate relative velocity and acceleration drift
            v_rel = v_C1 - v_C2
            a_drift_rel = a_drift_C1 - a_drift_C2
            J_rel = J_C1 - J_C2

            Lf_h_val = Lf_h(p_rel, v_rel)
            psi_val = psi_func(h_val, Lf_h_val, current_gamma_js_val)
            Lf_psi_val = Lf_psi(v_rel, a_drift_rel, p_rel, Lf_h_val, current_gamma_js_val)
            Lg_psi_val_arm = Lg_psi(J_rel, p_rel)

            constraints_qp.append(Lg_psi_val_arm @ u_qp_js >= -Lf_psi_val - current_beta_js_val * psi_val)

    # Joint acceleration limits
    constraints_qp.append(u_qp_js >= ddq_min_arm)
    constraints_qp.append(u_qp_js <= ddq_max_arm)

    # Joint velocity limits (forward integration based on current velocity and desired acceleration)
    dq_current_arm_val = dq_full_curr[:NUM_ARM_JOINTS]
    # dq_min <= dq_current + ddq * dt
    constraints_qp.append(u_qp_js >= (dq_min_arm - dq_current_arm_val) / dt_val)
    constraints_qp.append(u_qp_js <= (dq_max_arm - dq_current_arm_val) / dt_val)

    # Joint position limits (forward integration twice)
    q_current_arm_val = q_full_curr[:NUM_ARM_JOINTS]
    # q_min <= q_current + dq_current * dt + 0.5 * ddq * dt^2
    constraints_qp.append(u_qp_js >= 2 * (q_min_arm - q_current_arm_val - dq_current_arm_val * dt_val) / (dt_val**2))
    constraints_qp.append(u_qp_js <= 2 * (q_max_arm - q_current_arm_val - dq_current_arm_val * dt_val) / (dt_val**2))

    problem = cp.Problem(cp.Minimize(cost), constraints_qp)
    ddq_safe_arm_val = np.zeros(NUM_ARM_JOINTS)
    qp_solved_successfully = False 

    try:
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=25000)
    except cp.error.SolverError:
        node_logger.warn(f"OSQP SolverError. Trying default solver.")
        try:
            problem.solve(verbose=False) # Fallback to default solver
        except Exception as e_alt_solve:
            qp_failed_flag = True
            node_logger.warn("OSQP SolverError. Attempting emergency stop.") 


    # --- Check problem status and u_qp_js.value for definitive QP failure ---
    if not qp_failed_flag and (problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE):
        if u_qp_js.value is not None:
            ddq_safe_arm_val = u_qp_js.value
            qp_solved_successfully = True
        else: 
            qp_failed_flag = True # Solved, but no value (shouldn't happen with OPTIMAL status)
            node_logger.warn("QP solved, but u_qp_js.value is None. Emergency stop activated.")
    else: 
        # If not optimal or inaccurate, it means it failed or is infeasible.
        qp_failed_flag = True 
        node_logger.warn(f"QP status {problem.status}. Emergency stop activated.")
    
    return np.array(ddq_safe_arm_val).flatten(), qp_solved_successfully


# --- HOCBF-QP RECOVERY Safety Filter ---
def solve_recovery_qp_js(dt_val, q_full_curr, dq_full_curr, ddq_nominal_arm_val, current_gamma_js_val, current_beta_js_val, d_margin, current_obstacles_list_sim, active_links_list, node_logger):
    """
    This QP is triggered when the standard QP results in very slow motion.
    It infers the desired EE direction from the ddq_nominal command.
    """
    u_qp_js = cp.Variable(NUM_ARM_JOINTS)

    # --- CONSTRAINTS ---
    # The safety and joint limit constraints are IDENTICAL to the original QP.
    constraints_qp = []

    # Pre-calculate all necessary kinematics once.
    ddq_full_zeros = np.zeros(model.nv)
    pin.forwardKinematics(model, data, q_full_curr, dq_full_curr, ddq_full_zeros)
    pin.computeJointJacobians(model, data, q_full_curr)
    pin.updateFramePlacements(model, data)

    # --- NEW OBJECTIVE (Derived from ddq_nominal) ---
    # 1. Get the End-Effector Jacobian
    J_ee_full = pin.getFrameJacobian(model, data, EE_FRAME_ID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J_ee_p_arm = J_ee_full[:3, :NUM_ARM_JOINTS]

    # 2. Calculate the intended Cartesian acceleration from the nominal command
    a_ee_nominal_dir = J_ee_p_arm @ ddq_nominal_arm_val

    # 3. Define the objective: Maximize the resulting EE acceleration along the intended direction.
    progress_objective = a_ee_nominal_dir.T @ J_ee_p_arm @ u_qp_js
    objective = cp.Maximize(progress_objective)

        # Iterate through each defined LINK and each obstacle
    for link_info in active_links_list:
        start_frame_id = link_info['start_frame_id']
        end_frame_id = link_info['end_frame_id']
        link_radius_val = link_info['radius']

        # Get 3D positions of the link's start and end points
        p1 = data.oMf[start_frame_id].translation
        p2 = data.oMf[end_frame_id].translation

        for obs_item in current_obstacles_list_sim:
            # --- 1. FFind the closest points between the two capsule segments ---
            c_robot, c_obs, t = get_closest_points_between_segments(
                p1, p2, obs_item['pose_start'], obs_item['pose_end']
            )
            
            p_rel = c_robot - c_obs
            h_val = h_func(p_rel, link_radius_val, obs_item['radius'], d_margin)
            if h_val < 0.07: # Only add constraints if h is small (indicating a risk of collision) [0.15m^2 * 3 = 0.0675]

                # --- 2. Calculate kinematics for the closest point on the ROBOT ---
                J_C, v_C, a_drift_C = get_point_kinematics(
                    link_info['start_frame_id'], link_info['end_frame_id'], t, dq_full_curr
                )

                # --- 3. Calculate relative velocity and acceleration drift ---
                # Assuming obstacle velocity is constant, so acceleration is zero
                v_rel = v_C - obs_item['velocity']
                a_drift_rel = a_drift_C # - a_obs_drift (which is 0)

                # --- 4. Use the new HOCBF functions for capsules ---
                Lf_h_val = Lf_h(p_rel, v_rel)
                psi_val = psi_func(h_val, Lf_h_val, current_gamma_js_val)
                Lf_psi_val = Lf_psi(v_rel, a_drift_rel, p_rel, Lf_h_val, current_gamma_js_val)
                Lg_psi_val_arm = Lg_psi(J_C, p_rel)

                constraints_qp.append(Lg_psi_val_arm @ u_qp_js >= -Lf_psi_val - current_beta_js_val * psi_val)

    # Iterate through self-collision pairs
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

        c_link1, c_link2, t = get_closest_points_between_segments(p1_1, p2_1, p1_2, p2_2)
        p_rel = c_link1 - c_link2
        h_val = h_func(p_rel, link_radius_val_1, link_radius_val_2)
        if h_val < 0.02:  # Only add constraints if h is small (indicating a risk of self-collision)
            # Calculate kinematics for the closest point on the first link
            J_C1, v_C1, a_drift_C1 = get_point_kinematics(
                start_frame_id_1, end_frame_id_1, t, dq_full_curr
            )
            J_C2, v_C2, a_drift_C2 = get_point_kinematics(
                start_frame_id_2, end_frame_id_2, t, dq_full_curr
            )
            # Calculate relative velocity and acceleration drift
            v_rel = v_C1 - v_C2
            a_drift_rel = a_drift_C1 - a_drift_C2
            J_rel = J_C1 - J_C2

            Lf_h_val = Lf_h(p_rel, v_rel)
            psi_val = psi_func(h_val, Lf_h_val, current_gamma_js_val)
            Lf_psi_val = Lf_psi(v_rel, a_drift_rel, p_rel, Lf_h_val, current_gamma_js_val)
            Lg_psi_val_arm = Lg_psi(J_rel, p_rel)

            constraints_qp.append(Lg_psi_val_arm @ u_qp_js >= -Lf_psi_val - current_beta_js_val * psi_val)

    # Joint acceleration limits
    constraints_qp.append(u_qp_js >= ddq_min_arm)
    constraints_qp.append(u_qp_js <= ddq_max_arm)

    # Joint velocity limits (forward integration based on current velocity and desired acceleration)
    dq_current_arm_val = dq_full_curr[:NUM_ARM_JOINTS]
    # dq_min <= dq_current + ddq * dt
    constraints_qp.append(u_qp_js >= (dq_min_arm - dq_current_arm_val) / dt_val)
    constraints_qp.append(u_qp_js <= (dq_max_arm - dq_current_arm_val) / dt_val)

    # Ensure the resulting motion is not in the opposite direction of the intent.
    constraints_qp.append(progress_objective >= 0)

    problem = cp.Problem(objective, constraints_qp)
    ddq_recovery_arm_val = None
    qp_solved_successfully = False
    try:
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
            if u_qp_js.value is not None:
                ddq_recovery_arm_val = u_qp_js.value
                qp_solved_successfully = True
    except Exception:
        qp_solved_successfully = False

    return np.array(ddq_recovery_arm_val).flatten() if qp_solved_successfully else None, qp_solved_successfully


class CbfControllerNode(Node):
    def __init__(self):
        super().__init__('cbf_controller_node')
        self.get_logger().info('CBF Controller Node has been started.')

        # --- Declare and Load Parameters ---
        self.declare_parameter('scenario_config_file', '')
        scenario_path = self.get_parameter('scenario_config_file').get_parameter_value().string_value
        
        # Load parameters directly from the provided YAML file
        if not scenario_path or not os.path.exists(scenario_path):
            self.get_logger().fatal("HOCBF Node requires 'scenario_config_file', but it was not provided or file does not exist. Shutting down.")
            self.destroy_node()
            return
            
        with open(scenario_path, 'r') as file:
            config = yaml.safe_load(file)
            
        hocbf_params = config.get('hocbf_controller', {}).get('ros__parameters', {})
        obstacle_params_from_config = config.get('obstacles', [])
        
        self.goal_ee_pos = np.array(hocbf_params.get('goal_ee_pos', [0.3, 0.0, 0.5])) # Default value
        self.initial_gamma_js = float(hocbf_params.get('gamma_js', 2.0))
        self.initial_beta_js = float(hocbf_params.get('beta_js', 3.0))
        self.initial_gamma_js = 2.0
        self.initial_beta_js = 3.0
        self.d_margin = float(hocbf_params.get('d_margin', 0.0))
        self.d_margin = max(0.005, self.d_margin)
        self.output_basename = hocbf_params.get('output_data_basename', 'unnamed_scenario')
        self.initial_ee_pos = None

        # --- Load termination parameters ---
        self.goal_tolerance = float(hocbf_params.get('goal_tolerance_m', 0.02)) # Default 2cm
        self.goal_settle_time_s = float(hocbf_params.get('goal_settle_time_s', 2.0)) # Must be at goal for 2s
        self.max_sim_duration_s = float(hocbf_params.get('max_sim_duration_s', 60.0)) # Timeout after 1 minute
        self.max_sim_duration_s = 60.0
        self.at_goal_timer = 0.0 # Time we have been within goal tolerance
        self.is_shutdown_initiated = False # Flag to prevent multiple shutdowns
        self.final_run_status = "NONE" # Default to TIMEOUT

        self.get_logger().info(f"--- HOCBF Controller Config for '{self.output_basename}' ---")
        
        # Create the internal representation of obstacles
        self.current_obstacles_list_sim = []
        for obs in obstacle_params_from_config:
            self.current_obstacles_list_sim.append({
                'pose_start': np.array(obs['pose_start']['position']),
                'pose_end': np.array(obs['pose_start']['position']),
                'radius': float(obs['size']['radius']),
                'velocity': np.array(obs['velocity']['linear'])
            })
            # In case of sphere pose_end = pose_start
            if 'pose_end' in obs:
                self.current_obstacles_list_sim[-1]['pose_end'] = np.array(obs['pose_end']['position'])

        # Subscription to joint states
        self.joint_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )
        self.joint_state_subscription

        # Publisher for joint position commands to the C++ controller
        self.joint_command_publisher = self.create_publisher(
            Float64MultiArray,
            '/joint_position_controller/external_commands',
            10
        )

        # Publisher for RViz markers
        self.marker_publisher_1 = self.create_publisher(
            Marker,
            '/visualization_marker_1', # Standard topic for RViz markers
            10
        )

        self.marker_publisher_2 = self.create_publisher(
            Marker,
            '/visualization_marker_2',
            10
        )

        self.marker_publisher_3 = self.create_publisher(
            Marker,
            '/visualization_marker_3',
            10
        )

        self.marker_publisher_4 = self.create_publisher(
            Marker,
            '/visualization_marker_4',
            10
        )

        self.bot_marker_publisher_1 = self.create_publisher(
            Marker,
            '/bot_marker_1',
            10
        )

        self.bot_marker_publisher_2 = self.create_publisher(
            Marker,
            '/bot_marker_2',
            10
        )

        self.bot_marker_publisher_3 = self.create_publisher(
            Marker,
            '/bot_marker_3',
            10
        )

        self.bot_marker_publisher_4 = self.create_publisher(
            Marker,
            '/bot_marker_4',
            10
        )

        self.bot_marker_publisher_5 = self.create_publisher(
            Marker,
            '/bot_marker_5',
            10
        )

        self.robot_joint_names = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
            'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
        ]
        self.num_arm_joints = len(self.robot_joint_names)

        # History for plotting ALL h and psi values (overall min)
        self.history_data_frames = []

        # --- Dynamic CBF parameters (initially global values) ---
        self.current_gamma_js = self.initial_gamma_js
        self.current_beta_js = self.initial_beta_js

        self.current_joint_positions = np.zeros(self.num_arm_joints)
        self.current_joint_velocities = np.zeros(self.num_arm_joints)
        self.received_first_joint_state = False

        # Pinocchio full state for calculations
        self.q_full_pin = np.zeros(model.nq)
        self.dq_full_pin = np.zeros(model.nv)

        # Control loop frequency (Hz) - needs to be fast enough for integration
        self.control_frequency = 50 # Hz
        self.dt = 1.0 / self.control_frequency # Time step for integration

        self.target = [0.307, 0.0, 0.487] # Initial target position for the end-effector in Cartesian space

        # Timer to trigger the control loop
        self.timer = self.create_timer(self.dt, self.control_loop)

        # Timer for publishing markers (can be slower than control loop)
        self.marker_publish_timer = self.create_timer(0.5, self.publish_markers) # Publish markers every 0.5 seconds


    def joint_states_callback(self, msg):
        if not self.received_first_joint_state:
            # Ensure correct joint order for initial setup
            for i, name in enumerate(self.robot_joint_names):
                if name not in msg.name:
                    self.get_logger().error(f"Joint '{name}' not found in received joint states. Cannot proceed.")
                    self.destroy_node() # Self-destruct if essential joints are missing
                    return
            self.received_first_joint_state = True
            
        current_positions = []
        current_velocities = []

        # Map by name to ensure correct ordering, even if msg.name order changes
        for joint_name in self.robot_joint_names:
            try:
                idx = msg.name.index(joint_name)
                current_positions.append(msg.position[idx])
                current_velocities.append(msg.velocity[idx] if idx < len(msg.velocity) else 0.0)
            except ValueError:
                self.get_logger().warn(f"Joint '{joint_name}' not found in current JointState message. This should not happen if initial check passed.")
                return # Skip this update if any required joint is missing

        self.current_joint_positions = np.array(current_positions)
        self.current_joint_velocities = np.array(current_velocities)
        
        # Also update Pinocchio's full state based on actual robot state
        self.q_full_pin[:self.num_arm_joints] = self.current_joint_positions
        self.dq_full_pin[:self.num_arm_joints] = self.current_joint_velocities


    def control_loop(self):
        if not self.received_first_joint_state or self.is_shutdown_initiated:
            self.get_logger().info("Control loop skipped: No joint state received yet or shutdown initiated.")
            return
        
        for obs in self.current_obstacles_list_sim:
            obs['pose_start'] = obs['pose_start'] + obs['velocity'] * self.dt

        q_arm_current = self.current_joint_positions
        dq_arm_current = self.current_joint_velocities

        min_h_current = float('inf')
        min_psi_current = float('inf')
        min_dist_current = float('inf')


        # --- Track maximum required gamma for dynamic adjustment ---
        max_gamma_required_for_h = -float('inf') # Initialize with negative infinity
        max_beta_required_for_psi = -float('inf') 

        # Ensure Pinocchio data is fresh before calculating h/psi
        pin.forwardKinematics(model, data, self.q_full_pin, self.dq_full_pin, np.zeros(model.nv))
        pin.updateFramePlacements(model, data)

        # --- LOOP for h, psi, and dynamic gamma/beta calculation for LINKS ---
        current_h_psi_values = {}
        for link_info in active_links:
            # Get link properties
            start_frame_id = link_info['start_frame_id']
            end_frame_id = link_info['end_frame_id']
            link_radius_val = link_info['radius']

            # Get 3D positions of the link's start and end points from pre-updated data
            p1 = data.oMf[start_frame_id].translation
            p2 = data.oMf[end_frame_id].translation

            for obs_idx, obs_item in enumerate(self.current_obstacles_list_sim):
                # --- 1. Find the closest points between the two capsule segments ---
                c_robot, c_obs, t = get_closest_points_between_segments(
                    p1, p2, obs_item['pose_start'], obs_item['pose_end']
                )

                J_C, v_C, a_drift_C = get_point_kinematics(
                    link_info['start_frame_id'], link_info['end_frame_id'], t, self.dq_full_pin
                )

                # --- Calculate and track minimum surface distance ---
                dist_centers = np.linalg.norm(c_robot - c_obs)
                dist_surfaces = dist_centers - (link_radius_val + obs_item['radius'])
                min_dist_current = min(min_dist_current, dist_surfaces)

                # --- 2. Calculate h, Lf_h, and update min_h ---
                p_rel = c_robot - c_obs
                h_val = h_func(p_rel, link_radius_val, obs_item['radius'], self.d_margin)
                min_h_current = min(min_h_current, h_val)
                
                v_rel = v_C - obs_item['velocity']
                a_drift_rel = a_drift_C # - a_obs_drift (which is 0)
                Lf_h_val = Lf_h(p_rel, v_rel)

                # --- 3. Calculate required gamma ---
                if h_val > EPSILON_DENOMINATOR:
                    gamma_needed_for_pair = -Lf_h_val / h_val
                    max_gamma_required_for_h = max(max_gamma_required_for_h, gamma_needed_for_pair)
                elif h_val < -EPSILON_DENOMINATOR:
                    max_gamma_required_for_h = GAMMA_MAX_LIMIT

                # --- 4. Calculate psi and update min_psi (using the dynamically adjusted gamma for THIS step) ---
                psi_val = psi_func(h_val, Lf_h_val, self.current_gamma_js)
                min_psi_current = min(min_psi_current, psi_val)

                # --- 5. Calculate required beta ---
                Lf_psi_val = Lf_psi(v_rel, a_drift_rel, p_rel, Lf_h_val, self.current_gamma_js)
                Lg_psi_val = Lg_psi(J_C, p_rel).flatten()
                
                sup_Lg_psi_u = np.sum(np.where(Lg_psi_val >= 0, Lg_psi_val * ddq_max_arm, Lg_psi_val * ddq_min_arm))
                S_sup_val = Lf_psi_val + sup_Lg_psi_u

                if psi_val > EPSILON_DENOMINATOR:
                    beta_needed_for_pair = -S_sup_val / psi_val
                    max_beta_required_for_psi = max(max_beta_required_for_psi, beta_needed_for_pair)
                elif psi_val < -EPSILON_DENOMINATOR:
                    max_beta_required_for_psi = BETA_MAX_LIMIT

                # --- 6. Store individual h and psi values for logging/plotting ---
                key = f"{link_info['name']}_obs{obs_idx}"
                current_h_psi_values[f'h_{key}'] = h_val
                current_h_psi_values[f'psi_{key}'] = psi_val

        # --- Dynamically adjust gamma_js (This block is mostly the same, just follows the new loop) ---
        if max_gamma_required_for_h > -float('inf'):
            adjusted_gamma = max(0.0, max_gamma_required_for_h + GAMMA_ADJUST_BUFFER)
            self.current_gamma_js = max(self.initial_gamma_js, min(adjusted_gamma, GAMMA_MAX_LIMIT))
        # else: we keep the current_gamma_js

        # --- Dynamically adjust beta_js (This block is mostly the same) ---
        if max_beta_required_for_psi > -float('inf'):
            adjusted_beta = max(0.0, max_beta_required_for_psi + BETA_ADJUST_BUFFER)
            self.current_beta_js = max(self.initial_beta_js, min(adjusted_beta, BETA_MAX_LIMIT))
        # else: we keep the current_beta_js


        # Record current time
        current_time_ns = self.get_clock().now().nanoseconds
        if not self.history_data_frames: # First time step
            self.start_time_ns = current_time_ns
        current_time_s = (current_time_ns - self.start_time_ns) / 1e9

        # --- Check for termination conditions at the end of the loop ---
        # Get current EE position from Pinocchio data (already updated in your loop)
        current_ee_pos = data.oMf[EE_FRAME_ID].translation
        distance_to_goal = np.linalg.norm(current_ee_pos - self.goal_ee_pos)

        if self.initial_ee_pos is None:
            # Initialize the initial EE position on the first run
            self.initial_ee_pos = current_ee_pos.copy()

        # Condition 1: Goal Reached
        if distance_to_goal < self.goal_tolerance:
            self.at_goal_timer += self.dt
            if self.at_goal_timer >= self.goal_settle_time_s:
                self.get_logger().info(f"SUCCESS: Goal reached for {self.goal_settle_time_s}s. Simulation time {current_time_s:.2f}s.")
                self.final_run_status = "SUCCESS" # Set success status
                self.is_shutdown_initiated = True
                self.destroy_node() # This will trigger the main `finally` block
                return
        else:
            self.at_goal_timer = 0.0 # Reset timer if we move out of tolerance

        # Condition 2: Timeout
        if current_time_s > self.max_sim_duration_s:
            self.get_logger().warn(f"TIMEOUT: Simulation exceeded {self.max_sim_duration_s}s. Shutting down.")
            if np.linalg.norm(current_ee_pos - self.initial_ee_pos) < 0.01: # e.g., less than 1cm of movement
                self.get_logger().error("CONTROLLER FAILED: Robot did not move from initial position.")
                self.final_run_status = "FAILED_NO_MOVEMENT"
            else:
                self.final_run_status = "TIMEOUT"
            self.is_shutdown_initiated = True
            self.destroy_node()
            return

        start_solve_time = timer.time()

        ddq_nominal_arm_cmd, self.target = nominal_controller_js(q_arm_current, dq_arm_current, self.goal_ee_pos, self.target)
        # ddq_nominal_arm_cmd = nominal_controller_mpc(q_arm_current, dq_arm_current, self.goal_ee_pos, self.dt)

        
        # Safety filter (CBF-QP)
        # Pass the node's logger to the QP solver for better logging
        ddq_safe_arm_cmd, qp_solved = solve_hocbf_qp_js(
            self.dt, self.q_full_pin, self.dq_full_pin, ddq_nominal_arm_cmd, self.current_gamma_js,
            self.current_beta_js, self.d_margin, self.current_obstacles_list_sim, active_links, self.get_logger()
        )
        

        # Use nominal controller
        # qp_solved = True
        # ddq_safe_arm_cmd = ddq_nominal_arm_cmd

        '''
        if qp_solved:
            ddq_nominal_clipped = np.clip(ddq_nominal_arm_cmd, ddq_min_arm, ddq_max_arm)
            difference_norm = np.linalg.norm(ddq_nominal_clipped) - np.linalg.norm(ddq_safe_arm_cmd)
            if difference_norm > ACCEL_DIF_THRESHOLD:
                
                ddq_recovery_cmd, recovery_solved = solve_recovery_qp_js(
                    self.dt, self.q_full_pin, self.dq_full_pin, ddq_nominal_arm_cmd,
                    self.current_gamma_js, self.current_beta_js, self.d_margin,
                    self.current_obstacles_list_sim, active_links, self.get_logger()
                )

                if recovery_solved and np.linalg.norm(ddq_nominal_clipped) - np.linalg.norm(ddq_recovery_cmd) < difference_norm:
                    self.get_logger().info(f"Recovery QP successful. Applying performance-oriented command: {np.linalg.norm(ddq_recovery_cmd)}")
                    ddq_safe_arm_cmd = ddq_recovery_cmd
                else:
                    self.get_logger().warn(f"Recovery QP failed. Sticking to the original slow but safe command: {np.linalg.norm(ddq_safe_arm_cmd)}")
        '''
        
        
        if not qp_solved:
            # If QP was infeasible, we should not update the positions
            # Use the current positions instead
            next_dq_arm = dq_arm_current
            next_q_arm = q_arm_current
            self.get_logger().warn(f"QP infeasible, using current positions: {np.round(q_arm_current, 3)}")
        else:
            # Integrate joint dynamics (Euler) to get desired next positions
            next_dq_arm = dq_arm_current + ddq_safe_arm_cmd * self.dt
            next_q_arm = q_arm_current + dq_arm_current * self.dt + 0.5 * ddq_safe_arm_cmd * self.dt**2

        step_data = {
            'time': current_time_s,
            'solve_time': timer.time() - start_solve_time,
            'min_h': min_h_current,
            'min_dist': min_dist_current,
            'min_psi': min_psi_current,
            'qp_infeasible': not qp_solved,
            'joint_q': q_arm_current.tolist(),      # Store as list for compatibility
            'next_q': next_q_arm.tolist(),  # Store as list
            'joint_dq': dq_arm_current.tolist(),    # Store as list
            'next_dq': next_dq_arm.tolist(),  # Store as list
            'joint_ddq': ddq_safe_arm_cmd.tolist(),  # Store as list
            'joint_ddq_nominal': ddq_nominal_arm_cmd.tolist(),
            'current_gamma_js': self.current_gamma_js,
            'max_gamma_required_for_h': max_gamma_required_for_h, # Store the value that was calculated
            'current_beta_js': self.current_beta_js, # Fixed for now, but storing
            'max_beta_required_for_psi': max_beta_required_for_psi, # Store the value that was calculated
        }
        step_data.update(current_h_psi_values)

        self.history_data_frames.append(step_data)

        self.publish_joint_commands(next_q_arm)

    def publish_joint_commands(self, positions_array):
        msg = Float64MultiArray()
        msg.data = positions_array.tolist()
        self.joint_command_publisher.publish(msg)

    # --- Marker Publishing Functions ---
    def create_sphere_marker(self, marker_id, position, radius, r, g, b, a, frame_id="world"):
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
        marker.pose.orientation.w = 1.0 # No rotation for a sphere
        marker.scale.x = float(radius * 2) # Diameter
        marker.scale.y = float(radius * 2)
        marker.scale.z = float(radius * 2)
        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = float(a)
        marker.lifetime = rclpy.duration.Duration(seconds=0.7).to_msg() # Automatically disappear if not updated

        return marker
    
    def create_cylinder_marker(self, marker_id, p1, p2, radius, r, g, b, a, frame_id="world"):
        """Creates a CYLINDER marker for a link between two points."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "cbf_viz_links"
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        # Position is the midpoint of the link
        midpoint = (p1 + p2) / 2
        marker.pose.position.x = float(midpoint[0])
        marker.pose.position.y = float(midpoint[1])
        marker.pose.position.z = float(midpoint[2])

        # Scale is diameter (x,y) and length (z)
        link_vec = p2 - p1
        length = np.linalg.norm(link_vec)
        marker.scale.x = float(radius * 2)
        marker.scale.y = float(radius * 2)
        marker.scale.z = float(length)

        # Orientation: Calculate quaternion to align cylinder's Z-axis with the link vector
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
            else: # Aligned with Z-axis
                marker.pose.orientation.w = 1.0
        else: # Zero length
            marker.pose.orientation.w = 1.0

        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = float(a)
        marker.lifetime = rclpy.duration.Duration(seconds=0.7).to_msg()
        return marker

    def create_goal_marker(self, marker_id, position, frame_id="world"):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "cbf_viz"
        marker.id = marker_id
        marker.type = Marker.CUBE # Using a cube for goal
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
        marker.color.a = 0.8 # Green, semi-transparent
        marker.lifetime = rclpy.duration.Duration(seconds=0.7).to_msg()

        return marker

    def publish_markers(self):
        # Publish obstacle markers
        # Marker IDs for obstacles start from 0
        for i, obs in enumerate(self.current_obstacles_list_sim):
            marker_obs_cap1 = self.create_sphere_marker(
                marker_id=i,
                position=obs['pose_start'],
                radius=obs['radius'],
                r=1.0, g=0.5, b=0.0, a=0.7 # Orangered, semi-transparent
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
        # Marker ID for goal is after all obstacles
        marker_goal = self.create_goal_marker(
            marker_id=len(self.current_obstacles_list_sim)*3,
            position=self.goal_ee_pos
        )
        self.marker_publisher_1.publish(marker_goal)
        
        base_marker_id_link = len(self.current_obstacles_list_sim)*3 + 1 

        pin.forwardKinematics(model, data, self.q_full_pin, self.dq_full_pin, np.zeros(model.nv))
        pin.updateFramePlacements(model, data)

        bot_number_marker = 0
        for i, link_info in enumerate(active_links):
            p1 = data.oMf[link_info['start_frame_id']].translation
            p2 = data.oMf[link_info['end_frame_id']].translation
            
            marker_link = self.create_cylinder_marker(
                marker_id=base_marker_id_link + i,
                p1=p1,
                p2=p2,
                radius=link_info['radius'],
                r=0.2, g=0.2, b=0.8, a=0.6 # Blue, semi-transparent
            )
            marker_cap_1 = self.create_sphere_marker(
                marker_id=base_marker_id_link + len(active_links) + i, # Unique ID for each marker
                position=p1,
                radius=link_info['radius'],
                r=0.2, g=0.2, b=0.8, a=0.6 # Blue, semi-transparent for robot parts
            )
            marker_cap_2 = self.create_sphere_marker(
                marker_id=base_marker_id_link + 2*len(active_links) + i, # Unique ID for each marker
                position=p2,
                radius=link_info['radius'],
                r=0.2, g=0.2, b=0.8, a=0.6 # Blue, semi-transparent for robot parts
            )

            # Distribute markers among publishers (they don't plot well more than 5 markers at once)
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


    def save_and_plot_results(self):
        if not self.history_data_frames:
            self.get_logger().warn("No data to save or plot.")
            return
        
        # Define a directory to save plots (e.g., in your package)
        output_pack = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "sim")
        os.makedirs(output_pack, exist_ok=True) # Create directory if it doesn't exist
        output_dir = os.path.join(output_pack, self.output_basename)
        os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist

        # --- Save raw data to CSV using pandas ---
        df = pd.DataFrame(self.history_data_frames)
        df['final_run_status'] = self.final_run_status

        csv_filename = os.path.join(output_dir, "run_data.csv")
        df.to_csv(csv_filename, index=False)
        

        # --- Extract all data from history_data_frames ---
        time_history = [d['time'] for d in self.history_data_frames]
        h_min_history = [d['min_h'] for d in self.history_data_frames]
        psi_min_history = [d['min_psi'] for d in self.history_data_frames]
        qp_infeasible_history = [d['qp_infeasible'] for d in self.history_data_frames]
        
        # Prepare per-joint data
        joint_q_history_extracted = {name: [] for name in self.robot_joint_names}
        next_q_history_extracted = {name: [] for name in self.robot_joint_names}
        joint_dq_history_extracted = {name: [] for name in self.robot_joint_names}
        next_dq_history_extracted = {name: [] for name in self.robot_joint_names}
        joint_ddq_history_extracted = {name: [] for name in self.robot_joint_names}
        joint_ddq_nominal_history_extracted = {name: [] for name in self.robot_joint_names}

        # --- Extract dynamic gamma/beta history ---
        current_gamma_js_history = [d['current_gamma_js'] for d in self.history_data_frames]
        max_gamma_required_for_h_history = [d['max_gamma_required_for_h'] for d in self.history_data_frames]
        current_beta_js_history = [d['current_beta_js'] for d in self.history_data_frames]
        max_beta_required_for_psi_history = [d['max_beta_required_for_psi'] for d in self.history_data_frames]

        # --- Extract solution times history ---
        solve_time_history = [d['solve_time'] for d in self.history_data_frames]
        
        # Dictionaries to store all individual h_kj and psi_kj time series
        all_h_kj_plots = {} 
        all_psi_kj_plots = {}

        # Get all unique keys for h_kj and psi_kj from the first data frame
        if self.history_data_frames:
            sample_data = self.history_data_frames[0]
            for key in sample_data.keys():
                if key.startswith('h_') and key not in ['h_min']: # Exclude 'h_min'
                    all_h_kj_plots[key] = []
                elif key.startswith('psi_') and key not in ['psi_min']: # Exclude 'psi_min'
                    all_psi_kj_plots[key] = []

        for data_point in self.history_data_frames:
            # Populate per-joint data
            for i, joint_name in enumerate(self.robot_joint_names):
                joint_q_history_extracted[joint_name].append(data_point['joint_q'][i])
                next_q_history_extracted[joint_name].append(data_point['next_q'][i])
                joint_dq_history_extracted[joint_name].append(data_point['joint_dq'][i])
                next_dq_history_extracted[joint_name].append(data_point['next_dq'][i])
                joint_ddq_history_extracted[joint_name].append(data_point['joint_ddq'][i])
                joint_ddq_nominal_history_extracted[joint_name].append(data_point['joint_ddq_nominal'][i])
            
            # Populate individual h_kj and psi_kj data
            for key in all_h_kj_plots.keys():
                all_h_kj_plots[key].append(data_point[key])
            for key in all_psi_kj_plots.keys():
                all_psi_kj_plots[key].append(data_point[key])


        # --- Plot overall min h and psi ---
        fig_overview = plt.figure(figsize=(10, 6))
        ax_overview = fig_overview.add_subplot(111) 
        
        ax_overview.plot(time_history, h_min_history, label='Min $h_{kj}$')
        ax_overview.plot(time_history, psi_min_history, label='Min $\psi_{kj}$')
        
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

        # Save fig_overview
        # Use a consistent name for all plots from this run
        base_filename = os.path.join(output_dir, "plot")
        fig_overview.savefig(f"{base_filename}_cbf_overview.png")

        # --- Plot individual joint data ---
        num_joints = len(self.robot_joint_names)
        fig_joint_data, axs_joint_data = plt.subplots(num_joints, 3, figsize=(15, 3 * num_joints), sharex=True)
        fig_joint_data.suptitle('Joint States and Accelerations Over Time', fontsize=16)

        q_range = q_max_arm - q_min_arm

        for i, joint_name in enumerate(self.robot_joint_names):
            axs_joint_data[i, 0].plot(time_history, joint_q_history_extracted[joint_name], label=f'{joint_name} ($q$)')
            axs_joint_data[i, 0].plot(time_history, next_q_history_extracted[joint_name], label=f'{joint_name} (Next $q$)', alpha=0.5)
            axs_joint_data[i, 0].axhline(q_max_arm[i], color='k', linestyle=':', alpha=0.7, label='Max Position' if i==0 else "")
            axs_joint_data[i, 0].axhline(q_min_arm[i], color='k', linestyle=':', alpha=0.7, label='Min Position' if i==0 else "")
            axs_joint_data[i, 0].set_ylabel('Position (rad)')
            axs_joint_data[i, 0].grid(True)
            if i == 0: axs_joint_data[i, 0].legend()

            axs_joint_data[i, 1].plot(time_history, joint_dq_history_extracted[joint_name], label=f'{joint_name} ($\dot{{q}}$)')
            axs_joint_data[i, 1].plot(time_history, next_dq_history_extracted[joint_name], label=f'{joint_name} (Next $\dot{{q}}$)', alpha=0.5)
            axs_joint_data[i, 1].axhline(dq_max_arm[i], color='k', linestyle=':', alpha=0.7, label='Max Vel' if i==0 else "")
            axs_joint_data[i, 1].axhline(dq_min_arm[i], color='k', linestyle=':', alpha=0.7, label='Min Vel' if i==0 else "")
            axs_joint_data[i, 1].set_ylabel('Velocity (rad/s)')
            axs_joint_data[i, 1].grid(True)
            if i == 0: axs_joint_data[i, 1].legend()

            axs_joint_data[i, 2].plot(time_history, joint_ddq_history_extracted[joint_name], label=f'{joint_name} ($\ddot{{q}}$ Applied)')
            axs_joint_data[i, 2].plot(time_history, joint_ddq_nominal_history_extracted[joint_name], label=f'{joint_name} ($\ddot{{q}}$ Nominal)', alpha=0.7)
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

        # Save fig_joint_data
        fig_joint_data.savefig(f"{base_filename}_joint_states.png")


        # --- Plot all individual h_kj values ---
        fig_all_h, ax_all_h = plt.subplots(figsize=(12, 8))
        ax_all_h.set_title('All $h_{kj}$ Values Over Time (Control Point-Obstacle Pairs)')
        ax_all_h.set_xlabel('Time (s)')
        ax_all_h.set_ylabel('$h_{kj}$ Value')
        ax_all_h.axhline(0, color='r', linestyle='--', label='Safety Boundary (0)')
        ax_all_h.grid(True)

        colors = plt.cm.get_cmap('tab10', len(all_h_kj_plots)) # Generate distinct colors
        for i, (key, h_values) in enumerate(all_h_kj_plots.items()):
            ax_all_h.plot(time_history, h_values, label=key, color=colors(i))
        ax_all_h.legend(loc='best', ncol=2, fontsize='small')
        ax_all_h.set_ylim(-0.3, 1)
        fig_all_h.tight_layout()

        # Save fig_all_h
        fig_all_h.savefig(f"{base_filename}_all_h_values.png")


        # --- Plot all individual psi_kj values ---
        fig_all_psi, ax_all_psi = plt.subplots(figsize=(12, 8))
        ax_all_psi.set_title('All $\psi_{kj}$ Values Over Time (Control Point-Obstacle Pairs)')
        ax_all_psi.set_xlabel('Time (s)')
        ax_all_psi.set_ylabel('$\psi_{kj}$ Value')
        ax_all_psi.axhline(0, color='r', linestyle='--', label='Safety Boundary (0)')
        ax_all_psi.grid(True)

        colors = plt.cm.get_cmap('tab10', len(all_psi_kj_plots)) # Use same colormap
        for i, (key, psi_values) in enumerate(all_psi_kj_plots.items()):
            ax_all_psi.plot(time_history, psi_values, label=key, color=colors(i))
        ax_all_psi.legend(loc='best', ncol=2, fontsize='small')
        ax_all_psi.set_ylim(-0.3, 1)
        fig_all_psi.tight_layout()

        # Save fig_all_psi
        fig_all_psi.savefig(f"{base_filename}_all_psi_values.png")


        # --- Plot dynamic gamma_js ---
        fig_gamma_dynamic, ax_gamma_dynamic = plt.subplots(figsize=(10, 6))
        ax_gamma_dynamic.plot(time_history, current_gamma_js_history, label='Applied $\gamma_{js}$')
        ax_gamma_dynamic.plot(time_history, max_gamma_required_for_h_history, '--', label='Max Required $\gamma_{js}$')
        ax_gamma_dynamic.set_xlabel('Time (s)')
        ax_gamma_dynamic.set_ylabel('$\gamma_{js}$ Value')
        ax_gamma_dynamic.set_title('Dynamic $\gamma_{js}$ Over Time')
        ax_gamma_dynamic.legend()
        ax_gamma_dynamic.grid(True)
        ax_gamma_dynamic.set_ylim(0, GAMMA_MAX_LIMIT * 1.1) # Adjusted Y-limit for gamma plot
        fig_gamma_dynamic.tight_layout()

        fig_gamma_dynamic.savefig(f"{base_filename}_dynamic_gamma.png")

        # --- Plot dynamic beta_js ---
        fig_beta_dynamic, ax_beta_dynamic = plt.subplots(figsize=(10, 6))
        ax_beta_dynamic.plot(time_history, current_beta_js_history, label='Applied $\\beta_{js}$')
        ax_beta_dynamic.plot(time_history, max_beta_required_for_psi_history, '--', label='Max Required $\\beta_{js}$')
        ax_beta_dynamic.set_xlabel('Time (s)')
        ax_beta_dynamic.set_ylabel('$\\beta_{js}$ Value')
        ax_beta_dynamic.set_title('Dynamic $\\beta_{js}$ Over Time')
        ax_beta_dynamic.legend()
        ax_beta_dynamic.grid(True)
        ax_beta_dynamic.set_ylim(0, BETA_MAX_LIMIT * 1.1) # Adjusted Y-limit for beta plot
        fig_beta_dynamic.tight_layout()

        fig_beta_dynamic.savefig(f"{base_filename}_dynamic_beta.png")

        # --- Plot solution times ---
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
        #plt.show()
        plt.close('all')

def main(args=None):
    rclpy.init(args=args)
    node = CbfControllerNode()
    try:
        # Loop and process callbacks as long as the ROS context is alive and our node hasn't signaled for shutdown.
        while rclpy.ok() and not node.is_shutdown_initiated:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # This block now runs upon clean shutdown from destroy_node() OR KeyboardInterrupt
        node.save_and_plot_results()
        if not node.is_shutdown_initiated:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()