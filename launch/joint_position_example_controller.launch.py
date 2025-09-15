#  Copyright (c) 2023 Franka Robotics GmbH
#  Modifications Copyright (c) 2025 Your Name/Company
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # ## Arguments section ##
    # This section is almost identical to the cartesian_impedance_controller.launch.py
    # It defines all the necessary parameters to connect to the real robot.
    robot_ip_parameter_name = 'robot_ip'
    arm_id_parameter_name = 'arm_id'
    load_gripper_parameter_name = 'load_gripper'
    use_fake_hardware_parameter_name = 'use_fake_hardware'
    fake_sensor_commands_parameter_name = 'fake_sensor_commands'
    use_rviz_parameter_name = 'use_rviz'
    
    # Added argument from your Gazebo launch file for the HOCBF node
    scenario_config_file_parameter_name = 'scenario_config_file'

    # Launch Configurations
    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    arm_id = LaunchConfiguration(arm_id_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)
    use_rviz = LaunchConfiguration(use_rviz_parameter_name)
    scenario_config_file = LaunchConfiguration(scenario_config_file_parameter_name)

    # ## franka.launch.py Inclusion ##
    # This is the core part that starts the robot driver and ros2_control hardware interfaces.
    # It's taken directly from the cartesian_impedance_controller.launch.py example.
    franka_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([PathJoinSubstitution(
            [FindPackageShare('franka_bringup'), 'launch', 'franka.launch.py'])]),
        launch_arguments={
            robot_ip_parameter_name: robot_ip,
            arm_id_parameter_name: arm_id,
            load_gripper_parameter_name: load_gripper,
            use_fake_hardware_parameter_name: use_fake_hardware,
            fake_sensor_commands_parameter_name: fake_sensor_commands,
            use_rviz_parameter_name: use_rviz
        }.items(),
    )

    # ## Controller Spawner ##
    # This node spawns your 'joint_position_controller'.
    # It replaces the spawner for the 'cartesian_impedance_controller'.
    joint_position_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_position_controller'],
        output='screen',
    )
    
    # ## HOCBF Node ##
    # This is your Python-based controller node from the Gazebo setup.
    # It receives the scenario configuration file as a parameter.
    hocbf_controller_node = Node(
        package='cbf_safety_filter',  # Make sure this is the correct package name
        executable='HOCBF.py',
        name='HOCBF',
        output='screen',
        parameters=[{'scenario_config_file': scenario_config_file}]
    )

    return LaunchDescription([
        # ## Launch Arguments Declarations ##
        DeclareLaunchArgument(
            robot_ip_parameter_name,
            default_value='192.168.1.200', # IMPORTANT: Change this to your robot's IP
            description='Hostname or IP address of the robot.'),
        DeclareLaunchArgument(
            arm_id_parameter_name,
            default_value='fr3',
            description='ID of the type of arm used. Supported values: panda, fr3, fp3'),
        DeclareLaunchArgument(
            use_rviz_parameter_name,
            default_value='true',
            description='Visualize the robot in Rviz'),
        DeclareLaunchArgument(
            use_fake_hardware_parameter_name,
            default_value='false',
            description='Use fake hardware (for testing without a robot)'),
        DeclareLaunchArgument(
            fake_sensor_commands_parameter_name,
            default_value='false',
            description="Fake sensor commands. Only valid when '{}' is true".format(
                use_fake_hardware_parameter_name)),
        DeclareLaunchArgument(
            load_gripper_parameter_name,
            default_value='true',
            description='Use Franka Gripper as an end-effector.'),
        DeclareLaunchArgument(
            scenario_config_file_parameter_name,
            description='Full path to the scenario YAML file for HOCBF.py'),
        
        # ## Launch Actions ##
        franka_bringup_launch,
        joint_position_controller_spawner,

        # Use an event handler to launch the HOCBF.py node only after the 
        # C++ position controller has been successfully loaded and activated.
        # This prevents the Python script from trying to publish commands before the
        # controller is ready to receive them.
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_position_controller_spawner,
                on_exit=[hocbf_controller_node]
            )
        ),
    ])