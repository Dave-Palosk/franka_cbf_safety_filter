#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Bool
import numpy as np
import pinocchio as pin
import os
import yaml

# --- Pinocchio Model Setup ---
URDF_FILENAME = "fr3_robot.urdf"
package_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF_PATH = os.path.join(package_directory, "include", "urdf", URDF_FILENAME)

if not os.path.exists(URDF_PATH):
    rclpy.logging.get_logger('nominal_controller_js_node').error(f"URDF file '{URDF_FILENAME}' not found at {URDF_PATH}.")
try:
    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
    print(f"JS Nominal Controller: Pinocchio model loaded from {URDF_PATH}")
except Exception as e:
    rclpy.logging.get_logger('nominal_controller_js_node').error(f"Error loading URDF: {e}")
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


def nominal_controller_js(q_arm_curr, dq_arm_curr, target_ee_pos_cartesian, target):
    """
    Joint-space PD controller to Cartesian goal with null-space joint limit avoidance.
    
    Args:
        q_arm_curr: Current joint positions (7,)
        dq_arm_curr: Current joint velocities (7,)
        target_ee_pos_cartesian: Goal end-effector position in Cartesian space [x, y, z]
        target: Previous target position for smooth interpolation
    
    Returns:
        ddq_nominal_arm: Nominal joint acceleration command (7,)
        target: Updated target position after interpolation
    """
    # Controller gains
    Kp_cart = 500.0
    Kd_cart = 50.0
    alpha = 0.01  # Target interpolation factor
    lambda_damp = 0.01  # Damping for pseudo-inverse
    Kp_joint_limit = 200.0  # Gain for joint limit avoidance
    activation_threshold = 0.9  # Start avoiding at 90% of joint range
    
    # Setup full state vectors for Pinocchio
    q_full_curr = np.zeros(model.nq)
    q_full_curr[:NUM_ARM_JOINTS] = q_arm_curr
    dq_full_curr = np.zeros(model.nv)
    dq_full_curr[:NUM_ARM_JOINTS] = dq_arm_curr
    
    # Perform kinematics
    pin.forwardKinematics(model, data, q_full_curr, dq_full_curr, np.zeros(model.nv))
    pin.computeJointJacobians(model, data, q_full_curr)
    pin.updateFramePlacements(model, data)
    
    # Get current end-effector position and Jacobian
    current_ee_pos = data.oMf[EE_FRAME_ID].translation
    J_ee_full = pin.getFrameJacobian(model, data, EE_FRAME_ID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J_ee_p_arm = J_ee_full[:3, :NUM_ARM_JOINTS]  # Position part only
    
    # Smooth target interpolation
    target = (1 - alpha) * target + alpha * target_ee_pos_cartesian
    
    # Calculate Cartesian errors
    error_pos_cart = target - current_ee_pos
    current_ee_vel_cart = J_ee_p_arm @ dq_arm_curr
    error_vel_cart = -current_ee_vel_cart
    
    # Desired Cartesian force
    f_desired_cart = Kp_cart * error_pos_cart + Kd_cart * error_vel_cart
    
    # Primary task: Cartesian space control via pseudo-inverse
    J_pseudo_inv = np.linalg.pinv(J_ee_p_arm, rcond=lambda_damp)
    ddq_primary_task = J_pseudo_inv @ f_desired_cart
    
    # Secondary task: Joint limit avoidance in null space
    q_mid = (q_max_arm + q_min_arm) / 2.0
    q_range = q_max_arm - q_min_arm
    ddq_secondary_task = np.zeros(NUM_ARM_JOINTS)
    
    for i in range(NUM_ARM_JOINTS):
        # Check if joint is near its limits
        if abs(q_arm_curr[i] - q_mid[i]) > (q_range[i] * activation_threshold / 2.0):
            # Calculate repulsive gradient pushing away from limits
            gradient = -2 * (q_arm_curr[i] - q_mid[i]) / (q_range[i]**2)
            ddq_secondary_task[i] = Kp_joint_limit * gradient
    
    # Project secondary task into null space
    I = np.identity(NUM_ARM_JOINTS)
    null_space_projector = I - (J_pseudo_inv @ J_ee_p_arm)
    ddq_secondary_projected = null_space_projector @ ddq_secondary_task
    
    # Combine tasks
    ddq_nominal_arm = ddq_primary_task + ddq_secondary_projected
    
    return ddq_nominal_arm, target


class JSNominalControllerNode(Node):
    def __init__(self):
        super().__init__('js_nominal_controller_node')
        self.get_logger().info('Joint-Space PD Nominal Controller Node started.')

        # Declare parameters
        self.declare_parameter('scenario_config_file', '')
        self.declare_parameter('goal_ee_pos', [0.3, 0.0, 0.5])
        self.declare_parameter('control_frequency', 50.0)
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
        
        control_freq = self.get_parameter('control_frequency').get_parameter_value().double_value
        self.dt = 1.0 / control_freq

        # State variables
        self.robot_joint_names = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
            'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
        ]
        self.current_joint_positions = np.zeros(NUM_ARM_JOINTS)
        self.current_joint_velocities = np.zeros(NUM_ARM_JOINTS)
        self.received_first_joint_state = False

        # Target interpolation state
        self.target = np.array([0.307, 0.0, 0.487])  # Initial target

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

        # Timer for control loop
        self.control_timer = self.create_timer(self.dt, self.control_loop)
        
        self.get_logger().info(f'JS PD Controller running at {control_freq} Hz')
        self.get_logger().info(f'Goal: {self.goal_ee_pos}, tolerance: {self.goal_tolerance}m')

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
            self.at_goal_timer += self.dt
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

    def control_loop(self):
        """Runs at 50 Hz to compute and publish nominal accelerations."""
        if not self.received_first_joint_state:
            self.get_logger().info('Waiting for joint states...', throttle_duration_sec=1.0)
            return
        
        # Check if goal is reached
        if self.check_goal_reached():
            # Publish zero acceleration
            msg = Float64MultiArray()
            msg.data = [0.0] * NUM_ARM_JOINTS
            self.acceleration_publisher.publish(msg)
            return
        
        # Compute nominal acceleration using JS PD controller
        ddq_nominal, self.target = nominal_controller_js(
            self.current_joint_positions,
            self.current_joint_velocities,
            self.goal_ee_pos,
            self.target
        )
        
        # Publish nominal acceleration
        msg = Float64MultiArray()
        msg.data = ddq_nominal.tolist()
        self.acceleration_publisher.publish(msg)
        
        # Log distance to goal periodically
        if self.current_ee_pos is not None:
            distance = np.linalg.norm(self.current_ee_pos - self.goal_ee_pos)
            self.get_logger().info(
                f'Distance to goal: {distance:.4f}m',
                throttle_duration_sec=2.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = JSNominalControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
