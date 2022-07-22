from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='wifi_package',
            executable='wifi_node',
            name='wifi'
        ),
        Node(
            package='motor_package',
            executable='motor_node',
            name='motor'
        ),
        Node(
            package='conveyor_package',
            executable='conveyor_node',
            name='conveyor',
        )
    ])
