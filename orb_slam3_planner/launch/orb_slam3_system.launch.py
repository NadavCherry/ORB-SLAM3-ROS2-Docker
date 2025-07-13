from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription


def generate_launch_description():


    launch_nodes = []

    for i in range(0, 3):
        robot_ns = f'robot_{i}'
        # Landmark Publisher
        launch_nodes.append(
            Node(
                package='orb_slam3_planner',
                executable='landmark_publisher_node',
                namespace=robot_ns,
                name=f'landmark_publisher_node_{i}',
                output='screen',
                parameters=[
                    {'robot_namespace': robot_ns},
                ]
            )
        )

        # Autonomous Explorer Node
        launch_nodes.append(
            Node(
                package='orb_slam3_planner',
                executable='autonomous_explorer_node',
                namespace=robot_ns,
                name=f'autonomous_explorer_node_{i}',
                output='screen',
                parameters=[
                    {'robot_namespace': robot_ns},
                ]
            )
        )

        # Visualizer Node
        launch_nodes.append(
            Node(
                package='orb_slam3_planner',
                executable='autonomous_explorer_visualizer',
                namespace=robot_ns,
                name=f'autonomous_explorer_visualizer_{i}',
                parameters=[
                    {'robot_namespace': robot_ns},
                ],
                output='screen'
            )
        )
    return LaunchDescription(launch_nodes)
