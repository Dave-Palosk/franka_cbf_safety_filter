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

#pragma once

#include <string>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include "franka_semantic_components/franka_robot_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp" // Include the message type

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace cbf_safety_filter {

    
class JointPositionController : public controller_interface::ControllerInterface {
 public:
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  // Add a subscriber for joint commands
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr joint_command_subscriber_;
  std::vector<double> desired_joint_positions_; // To store the latest commanded positions
  std::vector<double> smoothed_joint_positions_; // To store the smoothed positions
  std::mutex desired_joint_positions_mutex_; // Mutex for thread-safe access to desired_joint_positions_

  std::string arm_id_;
  bool is_gazebo_{false};
  std::string robot_description_;
  const int num_joints = 7;
  std::array<double, 7> initial_q_{0, 0, 0, 0, 0, 0, 0};

  double alpha_{0.5}; // Default value

  const std::string k_HW_IF_INITIAL_POSITION = "initial_joint_position";

  bool initialization_flag_{true};
  
  // Callback for the joint command topic
  void joint_command_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
};

}  // namespace cbf_safety_filter