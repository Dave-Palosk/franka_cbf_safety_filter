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

#include <cbf_safety_filter/default_robot_behavior_utils.hpp>
#include <cbf_safety_filter/joint_velocity_example_controller.hpp>
#include <cbf_safety_filter/robot_utils.hpp>

#include <cassert>
#include <cmath>
#include <exception>
#include <string>

#include "std_msgs/msg/float64_multi_array.hpp"
#include <mutex>
#include <vector>

#include <Eigen/Eigen>

using namespace std::chrono_literals;

namespace cbf_safety_filter {
controller_interface::InterfaceConfiguration
JointVelocityExampleController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

controller_interface::InterfaceConfiguration
JointVelocityExampleController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}


CallbackReturn JointVelocityExampleController::on_init() {
  try {
    auto_declare<bool>("gazebo", false);
    auto_declare<std::string>("robot_description", "");
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

// Implement the joint command callback
void JointVelocityExampleController::joint_command_callback(
    const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
  // Lock the mutex to protect desired_joint_velocities_ during write
  std::lock_guard<std::mutex> lock(desired_joint_velocities_mutex_);
  if (msg->data.size() == num_joints) {
    desired_joint_velocities_ = msg->data;
  } else {
    RCLCPP_WARN(get_node()->get_logger(),
                "Received joint command with incorrect number of joints (%zu), expected %d. "
                "Ignoring command.", msg->data.size(), num_joints);
  }
}

// --- MODIFIED UPDATE FUNCTION ---
controller_interface::return_type JointVelocityExampleController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& /*period*/) {
  
  std::lock_guard<std::mutex> lock(desired_joint_velocities_mutex_);
  
  // Check if we have received a command yet.
  if (desired_joint_velocities_.size() != num_joints) {
    // If not, command zero velocity to be safe.
    for (int i = 0; i < num_joints; ++i) {
        command_interfaces_[i].set_value(0.0);
    }
    RCLCPP_DEBUG_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000, "Waiting for initial joint velocity commands...");
    return controller_interface::return_type::OK;
  }
  
  // Apply the received desired joint velocities.
  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(desired_joint_velocities_[i]);
  }
  
  return controller_interface::return_type::OK;
}
// --- END MODIFIED UPDATE FUNCTION ---


CallbackReturn JointVelocityExampleController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  is_gazebo = get_node()->get_parameter("gazebo").as_bool();

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

  // Initialize the subscriber for external velocity commands
  joint_command_subscriber_ = get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/joint_velocity_example_controller/commands", // Topic must match the publisher
      10,
      std::bind(&JointVelocityExampleController::joint_command_callback, this, std::placeholders::_1));

  RCLCPP_INFO(get_node()->get_logger(), "JointVelocityExampleController configured. Subscribing to /joint_velocity_example_controller/commands");

  if (!is_gazebo) {
    auto client = get_node()->create_client<franka_msgs::srv::SetFullCollisionBehavior>(
        "service_server/set_full_collision_behavior");
    auto request = DefaultRobotBehavior::getDefaultCollisionBehaviorRequest();

    auto future_result = client->async_send_request(request);
    future_result.wait_for(1000ms);

    auto success = future_result.get();
    if (!success) {
      RCLCPP_FATAL(get_node()->get_logger(), "Failed to set default collision behavior.");
      return CallbackReturn::ERROR;
    } else {
      RCLCPP_INFO(get_node()->get_logger(), "Default collision behavior set.");
    }
  }

  return CallbackReturn::SUCCESS;
}

// --- MODIFIED ON_ACTIVATE FUNCTION ---
CallbackReturn JointVelocityExampleController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
      
  // Initialize the desired velocities vector with zeros upon activation.
  // This prevents sudden movements before the first command is received.
  std::lock_guard<std::mutex> lock(desired_joint_velocities_mutex_);
  desired_joint_velocities_.resize(num_joints, 0.0);
  
  RCLCPP_INFO(get_node()->get_logger(), "JointVelocityExampleController activated. Initializing desired velocities to zero.");

  return CallbackReturn::SUCCESS;
}
// --- END MODIFIED ON_ACTIVATE FUNCTION ---

}  // namespace cbf_safety_filter
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(cbf_safety_filter::JointVelocityExampleController,
                       controller_interface::ControllerInterface)
