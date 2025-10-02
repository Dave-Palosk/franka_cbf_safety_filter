#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Bool
import numpy as np
import casadi as ca
import pinocchio as pin
import os
import yaml
import time as timer

# --- Pinocchio Model Setup (same as main file) ---
URDF_FILENAME = "fr3_robot.urdf"
package_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF_PATH = os.path.join(package_directory, "include", "urdf", URDF_FILENAME)

if not os.path.exists(URDF_PATH):
    rclpy.logging.get_logger('mpc_controller_node').error(f"URDF file '{URDF_FILENAME}' not found at {URDF_PATH}.")
try:
    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
    print(f"MPC Controller: Pinocchio model loaded from {URDF_PATH}")
except Exception as e:
    rclpy.logging.get_logger('mpc_controller_node').error(f"Error loading URDF: {e}")
    exit()

NUM_ARM_JOINTS = 7
EE_FRAME_NAME = "fr3_hand_tcp"
EE_FRAME_ID = model.getFrameId(EE_FRAME_NAME)

# Joint limits
ddq_max_scalar = 40.0
ddq_max_arm = np.full(NUM_ARM_JOINTS, ddq_max_scalar)
ddq_min_arm = np.full(NUM_ARM_JOINTS, -ddq_max_scalar)
dq_max_arm = np.array([2.0, 1.0, 1.5, 1.25, 3.0, 1.5, 3.0])
dq_min_arm = -dq_max_arm
q_max_arm = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
q_min_arm = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])


def nominal_controller_mpc(q_arm_curr, dq_arm_curr, target_ee_pos_cartesian, dt):
    """
    A simple kinematic MPC to generate a nominal joint acceleration command.
    """
    N = 7
    Q_pos = 100.0
    R_accel = 0.001
    W_jerk = 0.01

    opti = ca.Opti()
    ddq = opti.variable(NUM_ARM_JOINTS, N)
    q = opti.variable(NUM_ARM_JOINTS, N + 1)
    dq = opti.variable(NUM_ARM_JOINTS, N + 1)

    cost = 0
    cost += R_accel * ca.sumsqr(ddq)

    opti.subject_to(q[:, 0] == q_arm_curr)
    opti.subject_to(dq[:, 0] == dq_arm_curr)

    max_decel = np.abs(ddq_min_arm)
    stop_cost = 0
    jerk_cost = 0

    q_full_pin = np.zeros(model.nq)
    q_full_pin[:NUM_ARM_JOINTS] = q_arm_curr
    pin.forwardKinematics(model, data, q_full_pin)
    pin.computeJointJacobians(model, data, q_full_pin)
    pin.updateFramePlacements(model, data)
    
    J_ee_full = pin.getFrameJacobian(model, data, EE_FRAME_ID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J_ee_p_arm = J_ee_full[:3, :NUM_ARM_JOINTS]
    current_ee_pos = data.oMf[EE_FRAME_ID].translation

    for k in range(N):
        q_next = q[:, k] + dq[:, k] * dt + 0.5 * ddq[:, k] * dt**2
        dq_next = dq[:, k] + ddq[:, k] * dt

        opti.subject_to(q[:, k + 1] == q_next)
        opti.subject_to(dq[:, k + 1] == dq_next)
        opti.subject_to(opti.bounded(q_min_arm, q[:, k + 1], q_max_arm))
        opti.subject_to(opti.bounded(dq_min_arm, dq[:, k + 1], dq_max_arm))

        time_to_stop = (N - 1 - k) * dt
        dq_max_stoppable = max_decel * time_to_stop
        velocity_overshoot = ca.fmax(0, ca.fabs(dq[:, k+1]) - dq_max_stoppable)
        stop_cost += ca.sumsqr(velocity_overshoot)

        if k > 0:
            jerk_cost += ca.sumsqr(ddq[:, k] - ddq[:, k-1])

        predicted_ee_pos = current_ee_pos + J_ee_p_arm @ (q[:, k] - q_arm_curr)
        error_pos_cart = predicted_ee_pos - target_ee_pos_cartesian
        cost += Q_pos**(k/2) * ca.sumsqr(error_pos_cart)

    cost += W_jerk * jerk_cost
    opti.minimize(cost)

    p_opts = {"expand": True}
    s_opts = {"print_level": 0, "sb": "yes"}
    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol = opti.solve()
        planned_q = sol.value(q)
        planned_dq = sol.value(dq)
        return planned_q, planned_dq
    except RuntimeError:
        print("MPC Solver failed. Returning None.")
        return None, None


def joint_space_pid_controller(q_current, dq_current, q_target, dq_target):
    """
    A simple joint-space PD controller to track a target state.
    """
    Kp = 400.0
    Kd = 40.0
    
    error_pos = q_target - q_current
    error_vel = dq_target - dq_current
    
    ddq_nominal = Kp * error_pos + Kd * error_vel
    return ddq_nominal


class MPCPlannerNode(Node):
    def __init__(self):
        super().__init__('mpc_planner_node')
        self.get_logger().info('MPC Planner Node started.')

        # Declare parameters
        self.declare_parameter('scenario_config_file', '')
        self.declare_parameter('goal_ee_pos', [0.3, 0.0, 0.5])
        self.declare_parameter('mpc_frequency', 10.0)
        self.declare_parameter('pid_frequency', 50.0)
        self.declare_parameter('goal_tolerance_m', 0.02)
        self.declare_parameter('goal_settle_time_s', 2.0)
        
        # Load scenario file if provided
        scenario_path = self.get_parameter('scenario_config_file').get_parameter_value().string_value
        
        if scenario_path and os.path.exists(scenario_path):
            self.get_logger().info(f'Loading goal from scenario file: {scenario_path}')
            with open(scenario_path, 'r') as file:
                config = yaml.safe_load(file)
            hocbf_params = config.get('hocbf_controller', {}).get('ros__parameters', {})
            self.goal_ee_pos = np.array(hocbf_params.get('goal_ee_pos', [0.3, 0.0, 0.5]))
            self.goal_tolerance = float(hocbf_params.get('goal_tolerance_m', 0.02))
            self.goal_settle_time = float(hocbf_params.get('goal_settle_time_s', 2.0))
        else:
            # Fallback to parameters
            goal_param = self.get_parameter('goal_ee_pos').get_parameter_value().double_array_value
            self.goal_ee_pos = np.array(goal_param)
            self.goal_tolerance = self.get_parameter('goal_tolerance_m').get_parameter_value().double_value
            self.goal_settle_time = self.get_parameter('goal_settle_time_s').get_parameter_value().double_value
        
        mpc_freq = self.get_parameter('mpc_frequency').get_parameter_value().double_value
        pid_freq = self.get_parameter('pid_frequency').get_parameter_value().double_value
        
        self.mpc_dt = 1.0 / mpc_freq
        self.pid_dt = 1.0 / pid_freq

        # State variables
        self.robot_joint_names = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
            'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
        ]
        self.current_joint_positions = np.zeros(NUM_ARM_JOINTS)
        self.current_joint_velocities = np.zeros(NUM_ARM_JOINTS)
        self.received_first_joint_state = False

        # MPC trajectory storage
        self.mpc_planned_trajectory = None
        self.mpc_trajectory_start_time = 0.0
        self.mpc_solve_time = 0.0

        # Goal-reaching state
        self.at_goal_timer = 0.0
        self.goal_reached = False
        self.current_ee_pos = None

        # Subscription to joint states
        self.joint_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )

        # Publisher for nominal joint accelerations (to safety filter)
        self.acceleration_publisher = self.create_publisher(
            Float64MultiArray,
            '/nominal_joint_acceleration',
            10
        )
        
        # Publisher for goal status
        self.goal_status_publisher = self.create_publisher(
            Bool,
            '/goal_reached',
            10
        )

        # Timers
        self.mpc_timer = self.create_timer(self.mpc_dt, self.mpc_planning_loop)
        self.pid_timer = self.create_timer(self.pid_dt, self.pid_loop)
        
        # Store start time
        self.start_time_ns = self.get_clock().now().nanoseconds
        
        self.get_logger().info(f'MPC planning at {mpc_freq} Hz, PID tracking at {pid_freq} Hz')
        self.get_logger().info(f'Goal: {self.goal_ee_pos}, tolerance: {self.goal_tolerance}m, settle time: {self.goal_settle_time}s')

    def joint_states_callback(self, msg):
        if not self.received_first_joint_state:
            for name in self.robot_joint_names:
                if name not in msg.name:
                    self.get_logger().error(f"Joint '{name}' not found.")
                    return
            self.received_first_joint_state = True
            
        current_positions = []
        current_velocities = []

        for joint_name in self.robot_joint_names:
            try:
                idx = msg.name.index(joint_name)
                current_positions.append(msg.position[idx])
                current_velocities.append(msg.velocity[idx] if idx < len(msg.velocity) else 0.0)
            except ValueError:
                return

        self.current_joint_positions = np.array(current_positions)
        self.current_joint_velocities = np.array(current_velocities)
        
        # Update current EE position
        self.update_ee_position()

    def update_ee_position(self):
        """Calculate current end-effector position using forward kinematics."""
        q_full_pin = np.zeros(model.nq)
        q_full_pin[:NUM_ARM_JOINTS] = self.current_joint_positions
        pin.forwardKinematics(model, data, q_full_pin)
        pin.updateFramePlacements(model, data)
        self.current_ee_pos = data.oMf[EE_FRAME_ID].translation

    def check_goal_reached(self):
        """Check if the end-effector has reached and settled at the goal."""
        if self.current_ee_pos is None:
            return False
        
        distance_to_goal = np.linalg.norm(self.current_ee_pos - self.goal_ee_pos)
        
        if distance_to_goal <= self.goal_tolerance:
            self.at_goal_timer += self.pid_dt
            if self.at_goal_timer >= self.goal_settle_time:
                if not self.goal_reached:
                    self.goal_reached = True
                    self.get_logger().info(
                        f'GOAL REACHED! Distance: {distance_to_goal:.4f}m, '
                        f'Settled for {self.at_goal_timer:.2f}s'
                    )
                    # Publish goal reached status
                    msg = Bool()
                    msg.data = True
                    self.goal_status_publisher.publish(msg)
                return True
        else:
            self.at_goal_timer = 0.0
        
        return False

    def mpc_planning_loop(self):
        """Runs at 10 Hz to generate trajectory."""
        if not self.received_first_joint_state:
            return

        # Check if goal is reached - still plan but with zero motion
        if self.check_goal_reached():
            # Create stationary trajectory
            num_steps = 7
            current_q_repeated = np.tile(self.current_joint_positions, (num_steps, 1)).T
            current_dq_repeated = np.zeros((NUM_ARM_JOINTS, num_steps))
            self.mpc_planned_trajectory = {'q': current_q_repeated, 'dq': current_dq_repeated}
            self.mpc_solve_time = 0.0
            current_time_ns = self.get_clock().now().nanoseconds
            self.mpc_trajectory_start_time = (current_time_ns - self.start_time_ns) / 1e9
            return

        start_time = timer.time()

        planned_q, planned_dq = nominal_controller_mpc(
            self.current_joint_positions,
            self.current_joint_velocities,
            self.goal_ee_pos,
            self.mpc_dt
        )

        solve_time = timer.time() - start_time

        if planned_q is not None and planned_dq is not None:
            self.mpc_planned_trajectory = {'q': planned_q, 'dq': planned_dq}
            self.mpc_solve_time = solve_time
            
            # Store when this trajectory was generated
            current_time_ns = self.get_clock().now().nanoseconds
            self.mpc_trajectory_start_time = (current_time_ns - self.start_time_ns) / 1e9
            
            # Log distance to goal periodically
            if self.current_ee_pos is not None:
                distance = np.linalg.norm(self.current_ee_pos - self.goal_ee_pos)
                self.get_logger().info(
                    f'MPC solve time: {solve_time:.4f}s, distance to goal: {distance:.4f}m',
                    throttle_duration_sec=2.0
                )

    def pid_loop(self):
        """Runs at 50 Hz to track the MPC trajectory and publish accelerations."""
        if not self.received_first_joint_state:
            return
        
        if self.mpc_planned_trajectory is None:
            self.get_logger().info('Waiting for first MPC trajectory...', throttle_duration_sec=1.0)
            return
        
        # Get current time relative to start
        current_time_ns = self.get_clock().now().nanoseconds
        current_time_s = (current_time_ns - self.start_time_ns) / 1e9
        
        # 1. Determine the current time relative to the MPC plan's start time
        time_since_plan_start = current_time_s - self.mpc_trajectory_start_time

        # 2. Find the correct setpoint from the MPC's trajectory
        target_index = int(np.floor(time_since_plan_start / self.mpc_dt)) + 1
        
        # Ensure the index is within the bounds of the trajectory
        num_steps_in_plan = self.mpc_planned_trajectory['q'].shape[1]
        if target_index >= num_steps_in_plan:
            target_index = num_steps_in_plan - 1
        
        # Get target state from MPC trajectory
        q_target = self.mpc_planned_trajectory['q'][:, target_index]
        dq_target = self.mpc_planned_trajectory['dq'][:, target_index]
        # Calculate nominal acceleration using PID
        ddq_nominal = joint_space_pid_controller(
            self.current_joint_positions,
            self.current_joint_velocities,
            q_target,
            dq_target
        )
        
        # Publish nominal acceleration
        msg = Float64MultiArray()
        msg.data = ddq_nominal.tolist()
        self.acceleration_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MPCPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
