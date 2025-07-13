import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

robot_coordinates = {
    0: [-5.0, -7.0, 1.0],  # corridor
    1: [-1.0, 0.0, 1.65],
    2: [5.0, 5.0, 1.65],
    3: [-1.0, 8.0, 1.65],
    4: [7.0, 8.0, 1.65],
    5: [7.0, 8.0, 1.65],
}

def generate_launch_description():
    orb_slam3_launch_file_dir = os.path.join(
        get_package_share_directory('orb_slam3_ros2_wrapper'), 'launch')

    orb_slam3_launch_file_path = os.path.join(
            orb_slam3_launch_file_dir, 'rgbd.launch.py')
            
    # Spawn two ORB-SLAM3 instances
    orb_slam3_launch_description = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(orb_slam3_launch_file_path),
            launch_arguments={
                "robot_namespace": f"/robot_{i}/",
                "robot_x": str(robot_coordinates[i][0]),
                "robot_y": str(robot_coordinates[i][1]),
                "robot_z": str(robot_coordinates[i][2]),
            }.items(),
        )
        for i in range(3)
    ]

    return LaunchDescription(
        orb_slam3_launch_description
    )
