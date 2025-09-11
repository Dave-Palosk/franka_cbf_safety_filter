# Copyright (c) 2024 Franka Robotics GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import xacro

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit, OnShutdown

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch import LaunchContext, LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import  LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.actions import AppendEnvironmentVariable
from launch_ros.actions import Node
from launch.conditions import IfCondition

def get_robot_description(context: LaunchContext, arm_id, load_gripper, franka_hand, use_sim_time):
    print(f"\n[DEBUG] Loading robot configuration:")
    print(f"- arm_id: {context.perform_substitution(arm_id)}")
    print(f"- load_gripper: {context.perform_substitution(load_gripper)}")
    print(f"- franka_hand: {context.perform_substitution(franka_hand)}")

    arm_id_str = context.perform_substitution(arm_id)
    load_gripper_str = context.perform_substitution(load_gripper)
    franka_hand_str = context.perform_substitution(franka_hand)

    franka_xacro_file = os.path.join(
        get_package_share_directory('franka_description'),
        'robots',
        arm_id_str,
        arm_id_str + '.urdf.xacro'
    )

    print(f"[DEBUG] URDF file path: {franka_xacro_file}")
    if not os.path.exists(franka_xacro_file):
        print(f"[ERROR] URDF file not found at {franka_xacro_file}")
        raise FileNotFoundError(f"URDF file not found: {franka_xacro_file}")

    robot_description_config = xacro.process_file(
        franka_xacro_file, 
        mappings={
            'arm_id': arm_id_str, 
            'hand': load_gripper_str, 
            'ros2_control': 'true', 
            'gazebo': 'true', 
            'ee_id': franka_hand_str
        }
    )
    robot_description = {'robot_description': robot_description_config.toxml()}

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[
            robot_description,
            #{'use_sim_time': use_sim_time}
        ]
    )

    return [robot_state_publisher]


def prepare_launch_description():
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='False',
        description='Whether to run Gazebo in headless mode (no GUI).'
    )
    headless = LaunchConfiguration('headless')

    scenario_config_file_arg = DeclareLaunchArgument(
        'scenario_config_file',
        description='Full path to the scenario YAML file'
    )
    scenario_config_file = LaunchConfiguration('scenario_config_file')

    # Configure ROS nodes for launch
    load_gripper_name = 'load_gripper'
    franka_hand_name = 'franka_hand'
    arm_id_name = 'arm_id'

    load_gripper = LaunchConfiguration(load_gripper_name)
    franka_hand = LaunchConfiguration(franka_hand_name)
    arm_id = LaunchConfiguration(arm_id_name)

    load_gripper_launch_argument = DeclareLaunchArgument(
            load_gripper_name,
            default_value='false',
            description='true/false for activating the gripper')
    franka_hand_launch_argument = DeclareLaunchArgument(
            franka_hand_name,
            default_value='franka_hand',
            description='Default value: franka_hand')
    arm_id_launch_argument = DeclareLaunchArgument(
            arm_id_name,
            default_value='fr3',
            description='Available values: fr3, fp3 and fer')
    
    # Set use_sim_time for all nodes in the launch file
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Get robot description
    robot_state_publisher = OpaqueFunction(
        function=get_robot_description,
        args=[arm_id, load_gripper, franka_hand, use_sim_time])

    # Gazebo Sim
    
    # We now create the 'gz_args' dynamically based on the 'headless' value.
    # If headless is 'True', we add '-s' for server-only mode.
    gz_args_expression = PythonExpression([
        "'-r -s empty.sdf' if '", headless, "' == 'True' else '-r empty.sdf'"
    ])
    # for faster simulatoin use: ~/franka_ros2_ws/install/cbf_safety_filter/include/worlds/fast_world.sdf instead of empty.sdf in headless mode

    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    gazebo_empty_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': gz_args_expression}.items(),
    )

    # Spawn
    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', '/robot_description'],
        output='screen',
        #parameters=[{'use_sim_time': use_sim_time}]
    )

    # Visualize in RViz
    rviz_file = os.path.join(get_package_share_directory('franka_description'), 'rviz',
                             'visualize_franka.rviz')
    rviz = Node(package='rviz2',
             executable='rviz2',
             name='rviz2',
             arguments=['--display-config', rviz_file, '-f', 'world'],
             # Condition to launch only if headless is 'false'
             condition=IfCondition(PythonExpression(['not ', headless])),
             #parameters=[{'use_sim_time': use_sim_time}]
    )
    
    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
                'joint_state_broadcaster'],
        output='screen'
    )

    load_joint_position_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_position_controller'],
        output='screen'
    )

    # --- Pass the scenario file to the HOCBF Node ---
    hocbf_controller_node = Node(
        package='cbf_safety_filter',
        executable='HOCBF.py',
        name='HOCBF',
        output='screen',
        parameters=[
            #{'use_sim_time': use_sim_time},
            {'scenario_config_file': scenario_config_file}
        ]
    )
    set_env_vars_resources = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(get_package_share_directory('franka_description'))
    )

    kill_gazebo_on_shutdown = RegisterEventHandler(
        event_handler=OnShutdown(
            on_shutdown=[
                ExecuteProcess(
                    cmd=['pkill', '-f', 'ign gazebo'],
                    output='screen'
                )
            ]
        )
    )

    return LaunchDescription([
        # Add all arguments
        set_env_vars_resources,
        headless_arg,
        scenario_config_file_arg, # Add the new argument
        load_gripper_launch_argument,
        franka_hand_launch_argument,
        arm_id_launch_argument,
        use_sim_time_arg,
        
        # Add other actions and nodes
        gazebo_empty_world,
        robot_state_publisher,
        # rviz,
        spawn,

        RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=spawn,
                    on_exit=[load_joint_state_broadcaster],
                )
        ),    
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_state_broadcaster,
                on_exit=[load_joint_position_controller],
            )
        ),

        # Start the HOCBF controller only after the position controller is loaded and active.
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_position_controller,
                on_exit=[hocbf_controller_node]
            )
        ),

        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[
                #{'use_sim_time': use_sim_time},
                {'source_list': ['joint_states'], 'rate': 30}
            ],
        ),
        # hocbf_controller_node,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=hocbf_controller_node,
                on_exit=[
                    Shutdown(reason='Controller node exited')
                ],
            )
        ),
        kill_gazebo_on_shutdown
    ])

def generate_launch_description():
    launch_description = prepare_launch_description()

    set_env_vars_resources = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(get_package_share_directory('franka_description')))

    launch_description.add_action(set_env_vars_resources)

    return launch_description