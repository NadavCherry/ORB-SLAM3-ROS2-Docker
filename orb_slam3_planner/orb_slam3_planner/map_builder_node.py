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
        self.map_size = 40.0
        self.grid_size = int(self.map_size / self.resolution)
        self.grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        self.frame_count = 0
        self.rebuild_interval = 1  # Rebuild grid every 20 frames

        # Filter parameters
        self.z_min = 0.3  # Ignore floor points
        self.max_range = 40.0
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

    def save_grid_image(self):
        """Save occupancy grid as PNG image with better quality and auto-cropping"""
        # Create visualization array
        vis = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        vis[self.grid == 100] = 0  # Black = occupied
        vis[self.grid == -1] = 200  # Light gray = unknown (better contrast)
        vis = cv2.flip(vis, 0)  # Flip for proper orientation

        # Find the bounding box of occupied areas
        occupied_mask = (vis == 0)
        if np.any(occupied_mask):
            # Find all occupied pixels
            occupied_coords = np.where(occupied_mask)
            y_coords, x_coords = occupied_coords

            # Get tight bounding box
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()

            # Add smaller padding (since your map is dense)
            padding = 20
            y_min = max(0, y_min - padding)
            y_max = min(self.grid_size, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(self.grid_size, x_max + padding)

            # Crop to the tight area
            vis_cropped = vis[y_min:y_max, x_min:x_max]

            # Log the crop info for debugging
            print(f"Cropped from {self.grid_size}x{self.grid_size} to {vis_cropped.shape[0]}x{vis_cropped.shape[1]}")
        else:
            # If no occupied cells, use center area
            center = self.grid_size // 2
            margin = 50
            vis_cropped = vis[center - margin:center + margin, center - margin:center + margin]

        # Scale up the cropped image significantly
        scale_factor = 12  # Even bigger scaling since we're cropping tighter
        vis_large = cv2.resize(vis_cropped, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

        # Make occupied points bigger and more visible
        kernel = np.ones((4, 4), np.uint8)
        vis_large = cv2.dilate(vis_large, kernel, iterations=2)

        success1 = cv2.imwrite('/tmp/occupancy_grid_19_no_lines.png', vis_large)

        # # Add grid lines for reference (every meter)
        # self.add_grid_lines(vis_large, scale_factor, vis_cropped.shape)

        success2 = cv2.imwrite('/tmp/occupancy_grid_19.png', vis_large)
        if success1 and success2:
            self.get_logger().info("Saved occupancy grid to /tmp/occupancy_grid_18.png")
        else:
            self.get_logger().error("Failed to save image!")

    # def add_grid_lines(self, img, scale_factor, original_shape):
    #     """Add subtle grid lines every meter for reference"""
    #     height, width = img.shape
    #     # Grid spacing in original resolution (10 cells = 1 meter at 0.1m resolution)
    #     original_grid_spacing = 10
    #     grid_spacing = int(original_grid_spacing * scale_factor)
    #
    #     # Draw vertical lines
    #     for x in range(0, width, grid_spacing):
    #         cv2.line(img, (x, 0), (x, height - 1), 150, 1)
    #
    #     # Draw horizontal lines
    #     for y in range(0, height, grid_spacing):
    #         cv2.line(img, (0, y), (width - 1, y), 150, 1)


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