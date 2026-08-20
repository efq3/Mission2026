from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vtail_control',
            executable='mode_node',
            name='mode_node',
            output='screen'
        ),
        Node(
            package='vtail_control',
            executable='yolo_node',
            name='yolo_node',
            output='screen'
        ),
        Node(
            package='vtail_control',
            executable='guidance_node',
            name='guidance_node',
            output='screen'
        ),
        Node(
            package='vtail_control',
            executable='qgc_streamer_node',
            name='qgc_streamer_node',
            output='screen'
        ),
        Node(
            package='vtail_control',
            executable='Coral_PnP_xyz',
            name='Coral_PnP_xyz',
            output='screen'
        )
    ])
