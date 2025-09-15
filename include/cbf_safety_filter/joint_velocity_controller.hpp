#pragma once

#include <string>
#include <vector>
#include <mutex>

#include "controller_interface/controller_interface.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "rclcpp/subscription.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace cbf_safety_filter {

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class JointVelocityController : public controller_interface::ControllerInterface {
 public:
  JointVelocityController() = default;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::return_type update(const rclcpp::Time& time, const rclcpp::Duration& period) override;

  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  void joint_command_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

  const int num_joints = 7;
  std::string arm_id_;
  bool is_gazebo_ = false;
  std::string robot_description_;

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr joint_command_subscriber_;
  
  std::mutex desired_joint_velocities_mutex_;
  std::vector<double> desired_joint_velocities_;
};

}  // namespace cbf_safety_filter