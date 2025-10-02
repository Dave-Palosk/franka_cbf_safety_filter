from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    scenario_config_file_arg = DeclareLaunchArgument(
        'scenario_config_file',
        default_value='',
        description='Full path to the scenario YAML file (optional)'
    )
    
    mpc_frequency_arg = DeclareLaunchArgument(
        'mpc_frequency',
        default_value='10.0',
        description='MPC planning frequency in Hz'
    )
    
    pid_frequency_arg = DeclareLaunchArgument(
        'pid_frequency',
        default_value='50.0',
        description='PID tracking frequency in Hz'
    )
    
    mpc_planner_node = Node(
        package='cbf_safety_filter',
        executable='mpc_nominal_controller.py',
        name='mpc_planner',
        output='screen',
        parameters=[{
            'scenario_config_file': LaunchConfiguration('scenario_config_file'),
            'mpc_frequency': LaunchConfiguration('mpc_frequency'),
            'pid_frequency': LaunchConfiguration('pid_frequency')
        }]
    )
    
    return LaunchDescription([
        scenario_config_file_arg,
        mpc_frequency_arg,
        pid_frequency_arg,
        mpc_planner_node
    ])