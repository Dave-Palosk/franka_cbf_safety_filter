# Copyright (c) 2024 Franka Robotics GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License")

import os
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Simple launch file to spawn ONLY the joint_position_controller.
    Assumes Gazebo and controller_manager are already running.
    """
    
    joint_position_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_position_controller",
            "--controller-manager", "/controller_manager"
        ],
        output="screen",
    )
    
    return LaunchDescription([
        joint_position_controller_spawner
    ])