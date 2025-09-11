#include <cbf_safety_filter/joint_velocity_controller.hpp>
#include <cbf_safety_filter/robot_utils.hpp>

#include <string>
#include <vector>
#include <mutex>

#include "hardware_interface/types/hardware_interface_type_values.hpp"

namespace cbf_safety_filter {

controller_interface::InterfaceConfiguration
JointVelocityController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    // CRITICAL CHANGE: Request velocity interfaces to command
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/" + hardware_interface::HW_IF_VELOCITY);
  }
  return config;
}

controller_interface::InterfaceConfiguration
JointVelocityController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    // We still need position and velocity states for monitoring if needed,
    // though this simple controller doesn't use them in the update loop.
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/" + hardware_interface::HW_IF_POSITION);
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/" + hardware_interface::HW_IF_VELOCITY);
  }
  return config;
}

void JointVelocityController::joint_command_callback(
    const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(desired_joint_velocities_mutex_);
  if (msg->data.size() == num_joints) {
    desired_joint_velocities_ = msg->data;
  } else {
    RCLCPP_WARN(get_node()->get_logger(),
                "Received joint velocity command with incorrect size (%zu), expected %d.",
                msg->data.size(), num_joints);
  }
}

controller_interface::return_type JointVelocityController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& /*period*/) {
  std::lock_guard<std::mutex> lock(desired_joint_velocities_mutex_);

  if (desired_joint_velocities_.empty()) {
    return controller_interface::return_type::OK; // Do nothing if no command received
  }

  // Set the commanded velocity for each joint
  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(desired_joint_velocities_[i]);
  }

  return controller_interface::return_type::OK;
}

CallbackReturn JointVelocityController::on_init() {
  auto_declare<bool>("gazebo", false);
  auto_declare<std::string>("robot_description", "");
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointVelocityController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  is_gazebo_ = get_node()->get_parameter("gazebo").as_bool();

  auto parameters_client =
      std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "/robot_state_publisher");
  parameters_client->wait_for_service();
  auto future = parameters_client->get_parameters({"robot_description"});
  robot_description_ = future.get()[0].value_to_string();
  arm_id_ = robot_utils::getRobotNameFromDescription(robot_description_, get_node()->get_logger());

  // Subscribe to the new velocity command topic
  joint_command_subscriber_ = get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/joint_velocity_controller/commands", 10,
      std::bind(&JointVelocityController::joint_command_callback, this, std::placeholders::_1));

  RCLCPP_INFO(get_node()->get_logger(), "JointVelocityController configured.");
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointVelocityController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  std::lock_guard<std::mutex> lock(desired_joint_velocities_mutex_);
  // Initialize desired velocities to zero to prevent runaway motion on activation
  desired_joint_velocities_.assign(num_joints, 0.0);
  RCLCPP_INFO(get_node()->get_logger(), "JointVelocityController activated. Initializing commands to zero.");
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointVelocityController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  // On deactivation, command zero velocity to stop the robot
  for (int i = 0; i < num_joints; ++i) {
      command_interfaces_[i].set_value(0.0);
  }
  RCLCPP_INFO(get_node()->get_logger(), "JointVelocityController deactivated. Commanding zero velocity.");
  return CallbackReturn::SUCCESS;
}

}  // namespace cbf_safety_filter

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(cbf_safety_filter::JointVelocityController,
                       controller_interface::ControllerInterface)