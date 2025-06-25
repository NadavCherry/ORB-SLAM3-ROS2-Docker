from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    return LaunchDescription([
        Node(
            package='orb_slam3_planner',
            executable='filtered_landmark_publisher',
            name='filtered_landmark_publisher',
            output='screen'
        ),
        Node(
            package='orb_slam3_planner',
            executable='planner_node',
            name='orb_slam3_planner',
            output='screen'
        ),
    ])
