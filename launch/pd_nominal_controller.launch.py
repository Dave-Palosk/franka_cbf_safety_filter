from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    Launch file for Joint-Space PD Nominal Controller.
    This controller computes nominal joint accelerations at 50Hz.
    """
    
    scenario_config_file_arg = DeclareLaunchArgument(
        'scenario_config_file',
        default_value='',
        description='Full path to the scenario YAML file (optional)'
    )
    
    goal_ee_pos_arg = DeclareLaunchArgument(
        'goal_ee_pos',
        default_value='[0.3, 0.0, 0.5]',
        description='Goal end-effector position [x, y, z] (overridden by scenario file)'
    )
    
    control_frequency_arg = DeclareLaunchArgument(
        'control_frequency',
        default_value='50.0',
        description='Control loop frequency in Hz'
    )
    
    goal_tolerance_arg = DeclareLaunchArgument(
        'goal_tolerance_m',
        default_value='0.02',
        description='Distance tolerance for goal reaching (meters)'
    )
    
    goal_settle_time_arg = DeclareLaunchArgument(
        'goal_settle_time_s',
        default_value='2.0',
        description='Time to remain at goal before declaring success (seconds)'
    )
    
    js_controller_node = Node(
        package='cbf_safety_filter',
        executable='pd_nominal_controller.py',
        name='pd_nominal_controller',
        output='screen',
        parameters=[{
            'scenario_config_file': LaunchConfiguration('scenario_config_file'),
            'goal_ee_pos': LaunchConfiguration('goal_ee_pos'),
            'control_frequency': LaunchConfiguration('control_frequency'),
            'goal_tolerance_m': LaunchConfiguration('goal_tolerance_m'),
            'goal_settle_time_s': LaunchConfiguration('goal_settle_time_s')
        }]
    )
    
    return LaunchDescription([
        scenario_config_file_arg,
        goal_ee_pos_arg,
        control_frequency_arg,
        goal_tolerance_arg,
        goal_settle_time_arg,
        js_controller_node
    ])
