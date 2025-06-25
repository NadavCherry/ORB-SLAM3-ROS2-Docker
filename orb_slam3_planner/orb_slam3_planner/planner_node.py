#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import random
import threading


class OrbSlam3Planner(Node):
    def __init__(self):
        super().__init__('orb_slam3_planner')

        # Create QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid_2d', 1)

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/robot_pose_slam',
            self.pose_callback,
            10
        )

        # Subscribe to filtered landmarks instead of empty topics
        self.landmark_sub = self.create_subscription(
            PointCloud2,
            '/orb_slam3/landmarks_filtered',  # Use the filtered landmarks
            self.landmark_callback,
            qos_profile
        )

        # State variables
        self.current_pose = None
        self.latest_landmarks = None
        self.landmark_count = 0
        self.lock = threading.Lock()

        # Timer for main control loop
        self.timer = self.create_timer(1.0, self.step)

        self.get_logger().info('ORB-SLAM3 Planner initialized')
        self.get_logger().info(f'Listening for landmarks on: /orb_slam3/landmarks_filtered')
        self.get_logger().info(f'Listening for pose on: {self.pose_sub.topic_name}')

    def list_topics(self):
        """List all available topics to help debug"""
        topic_list = self.get_topic_names_and_types()
        self.get_logger().info('Available topics:')
        for topic_name, topic_types in topic_list:
            if 'PointCloud2' in str(topic_types) or 'PoseStamped' in str(topic_types):
                self.get_logger().info(f'  {topic_name}: {topic_types}')

        # Cancel this timer after first run
        self.destroy_timer(self.list_topics)

    def pose_callback(self, msg):
        """Updates current robot pose from ORB-SLAM3."""
        with self.lock:
            self.current_pose = msg
            self.get_logger().debug(f'Received pose: x={msg.pose.position.x:.2f}, '
                                    f'y={msg.pose.position.y:.2f}, z={msg.pose.position.z:.2f}')

    def landmark_callback(self, msg):
        """Receives landmarks from filtered publisher."""
        with self.lock:
            self.latest_landmarks = msg

            # Log information about received landmarks
            try:
                points = list(pc2.read_points(msg, skip_nans=True))
                self.landmark_count = len(points)

                if self.landmark_count > 0:
                    self.get_logger().info(f'Received {self.landmark_count} filtered landmarks')

                    # Process landmarks for occupancy grid
                    self.process_landmarks_to_occupancy_grid()

            except Exception as e:
                self.get_logger().error(f'Error processing landmarks: {e}')

    def process_landmarks_to_occupancy_grid(self):
        """Convert 3D landmarks to 2D occupancy grid"""
        if self.latest_landmarks is None:
            return None

        try:
            # Extract points from PointCloud2
            points = list(pc2.read_points(self.latest_landmarks, skip_nans=True))

            if len(points) == 0:
                return None

            # Convert to numpy array
            points_np = np.array([[p[0], p[1], p[2]] for p in points])

            # Create occupancy grid
            grid = OccupancyGrid()
            grid.header = self.latest_landmarks.header
            grid.info.resolution = 0.05  # 5cm resolution

            # Calculate grid bounds from points
            x_min, x_max = np.min(points_np[:, 0]), np.max(points_np[:, 0])
            y_min, y_max = np.min(points_np[:, 1]), np.max(points_np[:, 1])

            # Add margin
            margin = 2.0
            x_min -= margin
            x_max += margin
            y_min -= margin
            y_max += margin

            # Calculate grid dimensions
            grid.info.width = int((x_max - x_min) / grid.info.resolution)
            grid.info.height = int((y_max - y_min) / grid.info.resolution)
            grid.info.origin.position.x = x_min
            grid.info.origin.position.y = y_min
            grid.info.origin.position.z = 0.0

            # Initialize grid data (-1 = unknown)
            grid.data = [-1] * (grid.info.width * grid.info.height)

            # Project points to 2D and mark occupied cells
            for point in points_np:
                # Filter by height (e.g., only consider points between 0.1m and 1.5m)
                if 0.1 < point[2] < 1.5:
                    # Convert to grid coordinates
                    grid_x = int((point[0] - grid.info.origin.position.x) / grid.info.resolution)
                    grid_y = int((point[1] - grid.info.origin.position.y) / grid.info.resolution)

                    # Check bounds
                    if 0 <= grid_x < grid.info.width and 0 <= grid_y < grid.info.height:
                        index = grid_y * grid.info.width + grid_x
                        grid.data[index] = 100  # Mark as occupied

                        # Also mark neighboring cells for better visibility
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                nx, ny = grid_x + dx, grid_y + dy
                                if 0 <= nx < grid.info.width and 0 <= ny < grid.info.height:
                                    nindex = ny * grid.info.width + nx
                                    if grid.data[nindex] == -1:
                                        grid.data[nindex] = 100

            # Publish the occupancy grid
            self.map_pub.publish(grid)
            self.get_logger().debug(f'Published occupancy grid: {grid.info.width}x{grid.info.height}')

        except Exception as e:
            self.get_logger().error(f'Error creating occupancy grid: {e}')

    def step(self):
        """Main control loop"""
        with self.lock:
            if self.current_pose is None:
                self.get_logger().warn('No pose received yet')
                return

            if self.latest_landmarks is None:
                self.get_logger().warn('No landmarks received yet')
                return

            # Log current status
            self.get_logger().info(f'Planning with {self.landmark_count} landmarks')

            # TODO: Add your navigation logic here
            # Example: Simple reactive navigation
            cmd = Twist()

            # You can process the landmarks for obstacle avoidance
            # Linear velocity: 0.0 to 0.5 m/s
            cmd.linear.x = random.uniform(0.005, 0.1)

            # Angular velocity: random turn between -0.5 and 0.5 rad/s
            cmd.angular.z = random.uniform(-0.1, 0.1)
            self.cmd_pub.publish(cmd)

            self.get_logger().debug(f'Step executed - Pose: ({self.current_pose.pose.position.x:.2f}, '
                                    f'{self.current_pose.pose.position.y:.2f}), Landmarks: {self.landmark_count}')


def main(args=None):
    rclpy.init(args=args)

    planner = OrbSlam3Planner()

    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()