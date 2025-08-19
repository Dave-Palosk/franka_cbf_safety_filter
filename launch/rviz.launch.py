from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    rviz_file = os.path.join(get_package_share_directory('franka_description'), 'rviz',
                            'visualize_franka.rviz')
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['--display-config', rviz_file, '-f', 'world'],
    )

    return LaunchDescription([rviz_node])