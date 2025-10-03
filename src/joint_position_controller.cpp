// Copyright (c) 2023 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cbf_safety_filter/joint_position_controller.hpp>
#include <cbf_safety_filter/robot_utils.hpp>

#include <cassert>
#include <cmath>
#include <exception>
#include <string>
#include <mutex>
#include <vector>

#include <Eigen/Eigen>

namespace cbf_safety_filter {

controller_interface::InterfaceConfiguration
JointPositionController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
  }
  return config;
}

controller_interface::InterfaceConfiguration
JointPositionController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  if (!is_gazebo_) {
    for (int i = 1; i <= num_joints; ++i) {
      config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/" +
                             k_HW_IF_INITIAL_POSITION);
    }
  } else {
    for (int i = 1; i <= num_joints; ++i) {
      config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    }
  }
  return config;
}

// Implement the joint command callback
void JointPositionController::joint_command_callback(
    const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
  // Lock the mutex to protect desired_joint_positions_ during write
  std::lock_guard<std::mutex> lock(desired_joint_positions_mutex_);
  if (msg->data.size() == num_joints) {
    desired_joint_positions_ = msg->data;
  } else {
    RCLCPP_WARN(get_node()->get_logger(),
                "Received joint command with incorrect number of joints (%zu), expected %d. "
                "Ignoring command.", msg->data.size(), num_joints);
  }
}

controller_interface::return_type JointPositionController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& /*period*/) {

  // Lock the mutex to safely read desired_joint_positions_
  std::lock_guard<std::mutex> lock(desired_joint_positions_mutex_);
  if (desired_joint_positions_.empty()) {
    // If no command has been received yet, maintain current position or use initial_q_
    // For this example, let's just log a message and do nothing until a command arrives.

    RCLCPP_DEBUG_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000, "Waiting for initial joint commands...");
    return controller_interface::return_type::OK; // Do nothing if no command
  }


  // Apply the received desired joint positions through exponential smoothing filter
  for (int i = 0; i < num_joints; ++i) {
    // The formula: new = (1-alpha)*old + alpha*target
    // alpha defined in the header
    smoothed_joint_positions_[i] = (1 - alpha_) * smoothed_joint_positions_[i] + alpha_ * desired_joint_positions_[i];

    // Command the new, smoothed position to the robot
    command_interfaces_[i].set_value(smoothed_joint_positions_[i]);
  }

  return controller_interface::return_type::OK;
}

CallbackReturn JointPositionController::on_init() {
  try {
    auto_declare<bool>("gazebo", false);
    auto_declare<std::string>("robot_description", "");
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointPositionController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  is_gazebo_ = get_node()->get_parameter("gazebo").as_bool();

  auto parameters_client =
      std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "/robot_state_publisher");
  parameters_client->wait_for_service();

  auto future = parameters_client->get_parameters({"robot_description"});
  auto result = future.get();
  if (!result.empty()) {
    robot_description_ = result[0].value_to_string();
  } else {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
  }

  arm_id_ = robot_utils::getRobotNameFromDescription(robot_description_, get_node()->get_logger());

  // Initialize the subscriber here in on_configure
  // The topic name must match what your Python node publishes to.
  joint_command_subscriber_ = get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/joint_position_controller/external_commands",
      10,
      std::bind(&JointPositionController::joint_command_callback, this, std::placeholders::_1));

  RCLCPP_INFO(get_node()->get_logger(), "JointPositionController configured. Subscribing to /joint_position_controller/external_commands");

  return CallbackReturn::SUCCESS;
}

CallbackReturn JointPositionController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  // When activated, set desired_joint_positions_ to current actual positions
  // to avoid a sudden jump when the first external command is received.
  std::lock_guard<std::mutex> lock(desired_joint_positions_mutex_);
  desired_joint_positions_.resize(num_joints);
  smoothed_joint_positions_.resize(num_joints);
  for (int i = 0; i < num_joints; ++i) {
    const auto& initial_pos = state_interfaces_[i].get_value();
    desired_joint_positions_.at(i) = initial_pos;
    smoothed_joint_positions_.at(i) = initial_pos;
  }
  RCLCPP_INFO(get_node()->get_logger(), "JointPositionController activated. Initializing desired_joint_positions_ to current state.");

  return CallbackReturn::SUCCESS;
}

}  // namespace cbf_safety_filter
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(cbf_safety_filter::JointPositionController,
                       controller_interface::ControllerInterface)