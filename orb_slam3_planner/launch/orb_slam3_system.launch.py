from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    return LaunchDescription([
        Node(
            package='orb_slam3_planner',
            executable='landmark_publisher_node',
            name='landmark_publisher_node',
            output='screen'
        ),
        # Node(
        #     package='orb_slam3_planner',
        #     executable='map_builder_node',
        #     name='map_builder_node',
        #     output='screen'
        # ),
        # Node(
        #     package='orb_slam3_planner',
        #     executable='enhanced_map_builder_node',
        #     name='enhanced_map_builder_node',
        #     output='screen'
        # ),
        Node(
            package='orb_slam3_planner',
            executable='autonomous_explorer_node',
            name='autonomous_explorer_node',
            output='screen'
        ),
        Node(
            package='orb_slam3_planner',
            executable='autonomous_explorer_visualizer',
            name='autonomous_explorer_visualizer'
            # output='screen'
        ),
    ])
