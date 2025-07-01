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
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing


class EnhancedMapBuilderNode(Node):
    def __init__(self):
        super().__init__('enhanced_map_builder_node')

        # Grid parameters
        self.resolution = 0.1
        self.map_size = 40.0
        self.grid_size = int(self.map_size / self.resolution)

        # Two grids: raw accumulation and cleaned output
        self.point_count_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.occupancy_grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)

        self.frame_count = 0
        self.rebuild_interval = 50  # Process accumulated points every N frames

        # Filter parameters
        self.z_min = 0.3
        self.z_max = 2.0  # Also filter ceiling points
        self.max_range = 40.0
        self.min_points_for_occupied = 5  # Minimum points to consider cell occupied

        self.current_pose = None
        self.pose_lock = threading.Lock()

        # ROS setup
        self.create_subscription(PointCloud2, '/orb_slam3/landmarks_raw', self.cloud_callback, 10)
        self.create_subscription(PoseStamped, '/robot_pose_slam', self.pose_callback, 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)

        self.get_logger().info("Enhanced map builder started")

    def pose_callback(self, msg):
        with self.pose_lock:
            self.current_pose = msg.pose

    def cloud_callback(self, msg):
        with self.pose_lock:
            if self.current_pose is None:
                return
            robot_x = self.current_pose.position.x
            robot_y = self.current_pose.position.y

        # Accumulate points
        points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)

        for x, y, z in points:
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue

            # Height filtering
            if z < self.z_min or z > self.z_max:
                continue

            # Distance filtering
            distance = np.sqrt((x - robot_x) ** 2 + (y - robot_y) ** 2)
            if distance > self.max_range:
                continue

            # Convert to grid coordinates
            gx = int((x + self.map_size / 2) / self.resolution)
            gy = int((y + self.map_size / 2) / self.resolution)

            # Accumulate point count
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                self.point_count_grid[gy, gx] += 1

        self.frame_count += 1

        # Periodically process accumulated points and clean the map
        if self.frame_count % self.rebuild_interval == 0:
            self.process_and_clean_map()
            self.publish_grid()
            self.save_grid_image()

    def process_and_clean_map(self):
        """Convert point counts to clean occupancy grid"""
        # Create binary occupied/free map based on point density
        occupied_mask = self.point_count_grid >= self.min_points_for_occupied

        # Apply morphological operations to clean up the map
        # 1. Close small gaps in walls
        kernel_close = np.ones((3, 3), np.uint8)
        occupied_mask = binary_closing(occupied_mask, structure=kernel_close, iterations=2)

        # 2. Remove isolated noise points
        kernel_open = np.ones((2, 2), np.uint8)
        occupied_mask = binary_erosion(occupied_mask, structure=kernel_open, iterations=1)
        occupied_mask = binary_dilation(occupied_mask, structure=kernel_open, iterations=1)

        # 3. Slightly dilate walls for safety margin
        kernel_dilate = np.ones((3, 3), np.uint8)
        occupied_mask = binary_dilation(occupied_mask, structure=kernel_dilate, iterations=1)

        # Convert to occupancy grid format
        self.occupancy_grid.fill(-1)  # Unknown
        self.occupancy_grid[occupied_mask] = 100  # Occupied

        # Mark clearly free areas (areas with very few points near the robot)
        # This helps with navigation planning
        with self.pose_lock:
            if self.current_pose is not None:
                robot_gx = int((self.current_pose.position.x + self.map_size / 2) / self.resolution)
                robot_gy = int((self.current_pose.position.y + self.map_size / 2) / self.resolution)

                # Mark areas near robot with few points as free
                free_radius = int(5.0 / self.resolution)  # 5 meters
                for dy in range(-free_radius, free_radius + 1):
                    for dx in range(-free_radius, free_radius + 1):
                        gx = robot_gx + dx
                        gy = robot_gy + dy
                        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                            if self.point_count_grid[gy, gx] < 2 and not occupied_mask[gy, gx]:
                                self.occupancy_grid[gy, gx] = 0  # Free

        # Decay point counts to adapt to changes
        self.point_count_grid = (self.point_count_grid * 0.95).astype(np.int32)

    def publish_grid(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.resolution = self.resolution
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size
        msg.info.origin.position.x = -self.map_size / 2
        msg.info.origin.position.y = -self.map_size / 2
        msg.data = self.occupancy_grid.flatten().tolist()
        self.map_pub.publish(msg)

    def save_grid_image(self):
        """Save cleaned occupancy grid as PNG image"""
        # Create visualization
        vis = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        vis[self.occupancy_grid == 100] = [0, 0, 0]  # Black = occupied
        vis[self.occupancy_grid == 0] = [255, 255, 255]  # White = free
        vis[self.occupancy_grid == -1] = [200, 200, 200]  # Gray = unknown

        # Add robot position if available
        with self.pose_lock:
            if self.current_pose is not None:
                robot_gx = int((self.current_pose.position.x + self.map_size / 2) / self.resolution)
                robot_gy = int((self.current_pose.position.y + self.map_size / 2) / self.resolution)
                if 0 <= robot_gx < self.grid_size and 0 <= robot_gy < self.grid_size:
                    cv2.circle(vis, (robot_gx, robot_gy), 5, (0, 255, 0), -1)  # Green dot for robot

        vis = cv2.flip(vis, 0)

        # Auto-crop to relevant area
        occupied_mask = (vis[:, :, 0] == 0) | (vis[:, :, 0] == 255)  # Black or white pixels
        if np.any(occupied_mask):
            y_coords, x_coords = np.where(occupied_mask)
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()

            padding = 30
            y_min = max(0, y_min - padding)
            y_max = min(self.grid_size, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(self.grid_size, x_max + padding)

            vis_cropped = vis[y_min:y_max, x_min:x_max]
        else:
            center = self.grid_size // 2
            margin = 50
            vis_cropped = vis[center - margin:center + margin, center - margin:center + margin]

        # Scale up for better visibility
        scale_factor = 8
        vis_large = cv2.resize(vis_cropped, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

        cv2.imwrite('/tmp/enhanced_occupancy_grid.png', vis_large)
        self.get_logger().info("Saved enhanced occupancy grid to /tmp/enhanced_occupancy_grid.png")


def main(args=None):
    rclpy.init(args=args)
    node = EnhancedMapBuilderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()