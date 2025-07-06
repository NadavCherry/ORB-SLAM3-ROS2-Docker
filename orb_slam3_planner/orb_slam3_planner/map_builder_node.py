#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import threading
import cv2


class MapBuilderNode(Node):
    def __init__(self):
        super().__init__('map_builder_node')

        # Grid parameters
        self.resolution = 0.1
        self.map_size = 50.0
        self.grid_size = int(self.map_size / self.resolution)
        self.grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        self.frame_count = 0
        self.rebuild_interval = 1  # Rebuild grid every 20 frames

        # Filter parameters
        self.z_min = 0  # Ignore floor points
        self.max_range = 50.0
        self.current_pose = None
        self.pose_lock = threading.Lock()

        # ROS setup
        self.create_subscription(PointCloud2, '/orb_slam3/landmarks_raw', self.cloud_callback, 10)
        self.create_subscription(PoseStamped, '/robot_pose_slam', self.pose_callback, 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)

        self.get_logger().info("Planner started")

    def pose_callback(self, msg):
        with self.pose_lock:
            self.current_pose = msg.pose

    def cloud_callback(self, msg):
        with self.pose_lock:
            if self.current_pose is None:
                return  # Skip until we have robot position
            robot_x = self.current_pose.position.x
            robot_y = self.current_pose.position.y

        self.frame_count += 1

        # Periodically clear grid to stay in sync with SLAM
        if self.frame_count % self.rebuild_interval == 0:
            self.grid.fill(-1)

        points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)

        for x, y, z in points:
            # Skip invalid points
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue

            # Filter by height and distance from robot
            if z < self.z_min:
                continue

            distance = np.sqrt((x - robot_x) ** 2 + (y - robot_y) ** 2)
            if distance > self.max_range:
                continue

            # Convert to grid coordinates
            gx = int((x + self.map_size / 2) / self.resolution)
            gy = int((y + self.map_size / 2) / self.resolution)

            # Mark as occupied
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                self.grid[gy, gx] = 100

        self.publish_grid()
        self.save_grid_image()
        if self.frame_count % 20 == 0:
            self.save_grid_to_csv()

    def publish_grid(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.resolution = self.resolution
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size
        msg.info.origin.position.x = -self.map_size / 2
        msg.info.origin.position.y = -self.map_size / 2
        msg.data = self.grid.flatten().tolist()
        self.map_pub.publish(msg)

    def save_grid_to_csv(self, filename='/tmp/occupancy_grid.csv'):
        """
        Save the occupancy grid to a CSV file.

        Args:
            filename (str): Full path to the CSV file to save.
        """
        try:
            np.savetxt(filename, self.grid, fmt='%d', delimiter=',')
            self.get_logger().info(f"Saved occupancy grid to {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save occupancy grid to CSV: {e}")


    def save_grid_image(self):
        """Save occupancy grid as PNG image with better quality and auto-cropping"""
        vis = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        vis[self.grid == 100] = 255  # white = occupied
        vis[self.grid == -1] = 0  # black

        success1 = cv2.imwrite('/tmp/occupancy_grid_19.png', vis)
        success2 = cv2.imwrite('/tmp/occupancy_grid_19_grid.png', self.grid )
        if success1 and success2:
            self.get_logger().info("Saved occupancy grid to /tmp/occupancy_grid_18.png")
        else:
            self.get_logger().error("Failed to save image!")


def main(args=None):
    rclpy.init(args=args)
    node = MapBuilderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()