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
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os


class EnhancedMapBuilderNode(Node):
    def __init__(self):
        super().__init__('enhanced_map_builder_node')

        # Grid parameters for initial point cloud processing
        self.resolution = 0.1  # meters per pixel
        self.map_size = 50.0  # meters
        self.grid_size = int(self.map_size / self.resolution)

        # Planner grid parameters (larger cells for planning)
        self.planner_cell_size = 0.5  # Each planner cell is 0.5m x 0.5m
        self.planner_grid_resolution = int(self.planner_cell_size / self.resolution)  # 5 pixels per planner cell

        # Initialize grids
        self.raw_grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        self.planner_grid = None  # Will be created after processing

        self.frame_count = 0
        self.rebuild_interval = 20  # Rebuild grid every 20 frames

        # Filter parameters
        self.z_min = 0.1  # Ignore floor points
        self.z_max = 2.0  # Ignore ceiling points
        self.max_range = 20.0  # Maximum sensing range
        self.min_points_for_obstacle = 5  # Minimum points to consider a cell occupied

        # Robot state
        self.current_pose = None
        self.robot_grid_pos = None  # Robot position in planner grid coordinates
        self.pose_lock = threading.Lock()

        # ROS setup
        self.create_subscription(PointCloud2, '/orb_slam3/landmarks_raw', self.cloud_callback, 10)
        self.create_subscription(PoseStamped, '/robot_pose_slam', self.pose_callback, 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/planner_occupancy_grid', 10)

        # Visualization
        self.save_visualizations = True
        self.viz_path = '/tmp/planner_maps'
        os.makedirs(self.viz_path, exist_ok=True)

        self.get_logger().info("Enhanced Map Builder started")
        self.get_logger().info(f"Planner grid cell size: {self.planner_cell_size}m")

    def pose_callback(self, msg):
        """Update robot pose from SLAM"""
        with self.pose_lock:
            self.current_pose = msg.pose

    def cloud_callback(self, msg):
        """Process incoming point cloud and update occupancy grid"""
        with self.pose_lock:
            if self.current_pose is None:
                return
            robot_x = self.current_pose.position.x
            robot_y = self.current_pose.position.y

        self.frame_count += 1

        # Periodically clear grid to stay in sync with SLAM
        if self.frame_count % self.rebuild_interval == 0:
            self.raw_grid.fill(-1)

        # Process point cloud
        points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)

        # Track cells that have been observed (for free space detection)
        observed_cells = set()
        occupied_cells = set()

        for x, y, z in points:
            # Skip invalid points
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue

            # Filter by height
            if z < self.z_min or z > self.z_max:
                continue

            # Filter by distance from robot
            distance = np.sqrt((x - robot_x) ** 2 + (y - robot_y) ** 2)
            if distance > self.max_range:
                continue

            # Convert to grid coordinates
            gx = int((x + self.map_size / 2) / self.resolution)
            gy = int((y + self.map_size / 2) / self.resolution)

            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                occupied_cells.add((gx, gy))

            # Ray trace from robot to point to mark observed cells
            robot_gx = int((robot_x + self.map_size / 2) / self.resolution)
            robot_gy = int((robot_y + self.map_size / 2) / self.resolution)

            # Simple line drawing algorithm to trace ray
            cells_on_ray = self.bresenham_line(robot_gx, robot_gy, gx, gy)
            observed_cells.update(cells_on_ray[:-1])  # Don't include the endpoint (obstacle)

        # Update raw grid
        # Mark occupied cells
        for (gx, gy) in occupied_cells:
            self.raw_grid[gy, gx] = 100  # Occupied

        # Mark free cells (observed but not occupied)
        for (gx, gy) in observed_cells:
            if (gx, gy) not in occupied_cells and self.raw_grid[gy, gx] == -1:
                self.raw_grid[gy, gx] = 0  # Free

        # Process into planner grid
        self.process_planner_grid()

        # Publish and visualize
        self.publish_planner_grid()

        if self.frame_count % 10 == 0 and self.save_visualizations:
            self.save_grid_visualizations()

    def bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm for ray tracing"""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        while True:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                cells.append((x, y))

            if x == x1 and y == y1:
                break

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        return cells

    def bresenham_line_with_obstacles(self, x0, y0, x1, y1, grid):
        """Bresenham's line algorithm that stops at obstacles"""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        while True:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                cells.append((x, y))

                # Stop if we hit an obstacle (but include it in the ray)
                if grid[y, x] == 100 and (x, y) != (x0, y0):
                    break

            if x == x1 and y == y1:
                break

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        return cells

    def process_planner_grid(self):
        """Convert raw grid to planner grid with larger cells"""
        # Calculate planner grid dimensions
        planner_width = self.grid_size // self.planner_grid_resolution
        planner_height = self.grid_size // self.planner_grid_resolution

        # Initialize planner grid: -1=unknown, 0=free, 100=occupied
        self.planner_grid = np.full((planner_height, planner_width), -1, dtype=np.int8)

        # Process each planner cell
        for py in range(planner_height):
            for px in range(planner_width):
                # Get the corresponding region in raw grid
                start_x = px * self.planner_grid_resolution
                end_x = min((px + 1) * self.planner_grid_resolution, self.grid_size)
                start_y = py * self.planner_grid_resolution
                end_y = min((py + 1) * self.planner_grid_resolution, self.grid_size)

                # Extract the sub-region
                sub_region = self.raw_grid[start_y:end_y, start_x:end_x]

                # Count cell types
                occupied_count = np.sum(sub_region == 100)
                free_count = np.sum(sub_region == 0)
                unknown_count = np.sum(sub_region == -1)
                total_cells = sub_region.size

                # Determine planner cell status
                if occupied_count >= self.min_points_for_obstacle:
                    # If enough occupied cells, mark as obstacle
                    self.planner_grid[py, px] = 100
                elif free_count > total_cells * 0.5:
                    # If majority is free, mark as free
                    self.planner_grid[py, px] = 0
                elif unknown_count > total_cells * 0.8:
                    # If mostly unknown, keep as unknown
                    self.planner_grid[py, px] = -1
                else:
                    # Mixed case - be conservative and mark as free if some observation
                    self.planner_grid[py, px] = 0

        # Clean up the planner grid
        self.planner_grid = self.clean_planner_grid(self.planner_grid)

        # Update robot position in planner grid
        with self.pose_lock:
            if self.current_pose is not None:
                robot_x = self.current_pose.position.x
                robot_y = self.current_pose.position.y

                # Convert to planner grid coordinates
                pgx = int((robot_x + self.map_size / 2) / self.planner_cell_size)
                pgy = int((robot_y + self.map_size / 2) / self.planner_cell_size)

                if 0 <= pgx < planner_width and 0 <= pgy < planner_height:
                    self.robot_grid_pos = (pgx, pgy)
                    # Ensure robot position is marked as free
                    self.planner_grid[pgy, pgx] = 0

    def clean_planner_grid(self, grid):
        """Apply cleaning operations to the planner grid"""
        # Remove isolated occupied cells (likely noise)
        occupied_mask = (grid == 100)
        labeled, num_features = ndimage.label(occupied_mask)

        # Remove small occupied regions
        min_region_size = 2  # Minimum connected cells to keep
        for i in range(1, num_features + 1):
            region_size = np.sum(labeled == i)
            if region_size < min_region_size:
                grid[labeled == i] = 0  # Convert to free space

        # Optional: Apply morphological operations to smooth boundaries
        # kernel = np.ones((3, 3), np.uint8)
        # occupied_mask = (grid == 100).astype(np.uint8)
        # occupied_mask = cv2.morphologyEx(occupied_mask, cv2.MORPH_CLOSE, kernel)
        # grid[occupied_mask == 1] = 100

        return grid

    def publish_planner_grid(self):
        """Publish the planner grid as OccupancyGrid message"""
        if self.planner_grid is None:
            return

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        # Set resolution and dimensions for planner grid
        msg.info.resolution = self.planner_cell_size
        msg.info.width = self.planner_grid.shape[1]
        msg.info.height = self.planner_grid.shape[0]
        msg.info.origin.position.x = -self.map_size / 2
        msg.info.origin.position.y = -self.map_size / 2

        # Convert grid values to ROS occupancy grid format
        # ROS uses: 0-100 for occupied probability, -1 for unknown
        msg.data = self.planner_grid.flatten().tolist()

        self.map_pub.publish(msg)

    def save_grid_visualizations(self):
        """Save visualization of both raw and planner grids"""
        if self.planner_grid is None:
            return

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Raw grid
        ax1 = axes[0]
        raw_vis = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        raw_vis[self.raw_grid == 100] = [255, 0, 0]  # Red for occupied
        raw_vis[self.raw_grid == 0] = [0, 255, 0]  # Green for free
        raw_vis[self.raw_grid == -1] = [128, 128, 128]  # Gray for unknown

        ax1.imshow(raw_vis, origin='lower')
        ax1.set_title(f'Raw Grid ({self.resolution}m/cell)', fontsize=14)
        ax1.set_xlabel('X (cells)')
        ax1.set_ylabel('Y (cells)')

        # Plot 2: Planner grid
        ax2 = axes[1]
        planner_vis = np.zeros((*self.planner_grid.shape, 3), dtype=np.uint8)
        planner_vis[self.planner_grid == 100] = [255, 0, 0]  # Red for occupied
        planner_vis[self.planner_grid == 0] = [0, 255, 0]  # Green for free
        planner_vis[self.planner_grid == -1] = [128, 128, 128]  # Gray for unknown

        # Mark robot position
        if self.robot_grid_pos is not None:
            rx, ry = self.robot_grid_pos
            if 0 <= rx < self.planner_grid.shape[1] and 0 <= ry < self.planner_grid.shape[0]:
                planner_vis[ry, rx] = [0, 0, 255]  # Blue for robot

        ax2.imshow(planner_vis, origin='lower')
        ax2.set_title(f'Planner Grid ({self.planner_cell_size}m/cell)', fontsize=14)
        ax2.set_xlabel('X (cells)')
        ax2.set_ylabel('Y (cells)')

        # Add grid lines
        for i in range(self.planner_grid.shape[1] + 1):
            ax2.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
        for i in range(self.planner_grid.shape[0] + 1):
            ax2.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)

        # Plot 3: Planner grid with values
        ax3 = axes[2]
        im = ax3.imshow(self.planner_grid, origin='lower', cmap='RdYlGn_r', vmin=-1, vmax=100)
        ax3.set_title('Planner Grid Values', fontsize=14)
        ax3.set_xlabel('X (cells)')
        ax3.set_ylabel('Y (cells)')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Cell Value (-1=unknown, 0=free, 100=occupied)')

        # Add grid lines
        for i in range(self.planner_grid.shape[1] + 1):
            ax3.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
        for i in range(self.planner_grid.shape[0] + 1):
            ax3.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)

        plt.tight_layout()

        # Save figure
        filename = os.path.join(self.viz_path, f'planner_grid_frame_{self.frame_count:06d}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        # Also save individual grids
        # Save planner grid as numpy array
        np_filename = os.path.join(self.viz_path, 'planner_grid_latest.npy')
        np.save(np_filename, self.planner_grid)

        # Save planner grid as CSV for easy inspection
        csv_filename = os.path.join(self.viz_path, 'planner_grid_latest.csv')
        np.savetxt(csv_filename, self.planner_grid, fmt='%d', delimiter=',')

        # Save a simple image of just the planner grid
        planner_img = np.zeros((self.planner_grid.shape[0], self.planner_grid.shape[1]), dtype=np.uint8)
        planner_img[self.planner_grid == 100] = 255  # White for occupied
        planner_img[self.planner_grid == 0] = 128  # Gray for free
        planner_img[self.planner_grid == -1] = 0  # Black for unknown

        img_filename = os.path.join(self.viz_path, 'planner_grid_latest.png')
        cv2.imwrite(img_filename, planner_img)

        self.get_logger().info(f"Saved grid visualizations to {self.viz_path}")

        # Log grid statistics
        occupied_cells = np.sum(self.planner_grid == 100)
        free_cells = np.sum(self.planner_grid == 0)
        unknown_cells = np.sum(self.planner_grid == -1)
        total_cells = self.planner_grid.size

        self.get_logger().info(
            f"Planner Grid Stats - Occupied: {occupied_cells} ({occupied_cells / total_cells * 100:.1f}%), "
            f"Free: {free_cells} ({free_cells / total_cells * 100:.1f}%), "
            f"Unknown: {unknown_cells} ({unknown_cells / total_cells * 100:.1f}%)"
        )

        if self.robot_grid_pos:
            self.get_logger().info(f"Robot position in planner grid: {self.robot_grid_pos}")


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