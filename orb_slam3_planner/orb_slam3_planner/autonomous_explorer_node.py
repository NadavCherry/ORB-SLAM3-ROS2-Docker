#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import OccupancyGrid
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import math
from typing import Optional, Tuple, List


class AutonomousExplorerNode(Node):
    def __init__(self):
        super().__init__('autonomous_explorer_node')

        # Simplified parameters
        self.cell_size = 0.5  # meters per cell
        self.map_range = 25.0  # map extends ±25m
        self.grid_size = int(2 * self.map_range / self.cell_size)  # 100x100 grid

        # Robot state
        self.robot_pos = None  # (x, y) in grid coordinates
        self.robot_angle = 0.0  # current heading in radians
        self.current_pose = None

        # Simple grid (0=free, 1=obstacle, -1=unknown)
        self.grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)

        # Navigation state
        self.target = None
        self.state = "EXPLORING"  # EXPLORING, MOVING_TO_TARGET, TURNING
        self.last_update = self.get_clock().now()

        # Map cleaning parameters (like original)
        self.frame_count = 0
        self.rebuild_interval = 1  # Clear map every frame to stay in sync with SLAM
        self.min_points_for_obstacle = 3  # Need multiple points to confirm obstacle

        # Movement parameters
        self.linear_speed = 0.2
        self.angular_speed = 0.3
        self.safe_distance = 2  # cells

        # ROS setup
        self.create_subscription(PointCloud2, '/orb_slam3/landmarks_raw', self.pointcloud_callback, 10)
        self.create_subscription(PoseStamped, '/robot_pose_slam', self.pose_callback, 10)

        # Publish to your original topic names
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/planner_occupancy_grid', 10)
        self.goal_pub = self.create_publisher(Point, '/goal_grid_pos', 10)
        self.robot_pos_pub = self.create_publisher(Point, '/robot_grid_pos', 10)

        # Main control timer
        self.create_timer(0.5, self.control_loop)  # 2Hz

        self.get_logger().info("Autonomous Explorer Node started!")
        self.get_logger().info(f"Grid size: {self.grid_size}x{self.grid_size}, Cell size: {self.cell_size}m")

    def pose_callback(self, msg):
        """Update robot position from SLAM"""
        self.current_pose = msg.pose

        # Convert to grid coordinates
        world_x = msg.pose.position.x
        world_y = msg.pose.position.y

        grid_x = int((world_x + self.map_range) / self.cell_size)
        grid_y = int((world_y + self.map_range) / self.cell_size)

        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
            self.robot_pos = (grid_x, grid_y)

            # Publish robot position
            robot_msg = Point()
            robot_msg.x = float(grid_x)
            robot_msg.y = float(grid_y)
            robot_msg.z = float(self.get_robot_direction())  # Direction as z component
            self.robot_pos_pub.publish(robot_msg)

        # Extract heading angle
        q = msg.pose.orientation
        self.robot_angle = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def pointcloud_callback(self, msg):
        """Simple mapping from point cloud with periodic cleaning"""
        if self.current_pose is None:
            return

        robot_world_x = self.current_pose.position.x
        robot_world_y = self.current_pose.position.y

        self.frame_count += 1

        # Periodically clear grid to stay in sync with SLAM (like original)
        if self.frame_count % self.rebuild_interval == 0:
            self.grid.fill(-1)  # Reset to unknown
            self.get_logger().debug(f"Map cleared at frame {self.frame_count}")

        # Clear area around robot (assume it's free)
        if self.robot_pos:
            rx, ry = self.robot_pos
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = rx + dx, ry + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        self.grid[ny, nx] = 0

        # Process point cloud
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

        # First pass: collect and count points per cell
        cell_point_counts = {}
        valid_points = []

        for x, y, z in points:
            # Filter points by height (obstacles on ground level)
            if not (0.1 <= z <= 2.0):
                continue

            # Check if point is finite
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue

            # Convert to grid
            grid_x = int((x + self.map_range) / self.cell_size)
            grid_y = int((y + self.map_range) / self.cell_size)

            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                cell_key = (grid_x, grid_y)
                cell_point_counts[cell_key] = cell_point_counts.get(cell_key, 0) + 1
                valid_points.append((grid_x, grid_y))

        # Mark cells as obstacles only if they have enough points
        occupied_cells = set()
        for (gx, gy), count in cell_point_counts.items():
            if count >= self.min_points_for_obstacle:
                self.grid[gy, gx] = 1  # Mark as obstacle
                occupied_cells.add((gx, gy))

        # Second pass: ray trace to mark free cells (only to confirmed obstacles)
        if self.robot_pos:
            robot_gx, robot_gy = self.robot_pos
            observed_cells = set()

            for (gx, gy) in occupied_cells:  # Only trace to confirmed obstacles
                cells_on_ray = self.mark_free_path_advanced(robot_gx, robot_gy, gx, gy)
                observed_cells.update(cells_on_ray[:-1])  # Exclude the obstacle cell itself

            # Mark free cells (only if they were unknown)
            for (gx, gy) in observed_cells:
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    if self.grid[gy, gx] == -1:  # Only mark unknown cells as free
                        self.grid[gy, gx] = 0

        # Publish map
        self.publish_map()

    def get_robot_direction(self):
        """Convert robot angle to discrete direction (0=North, 1=East, 2=South, 3=West)"""
        # Normalize angle to [0, 2π]
        angle = self.robot_angle
        if angle < 0:
            angle += 2 * math.pi

        # Convert to 4 directions
        # 0 = North (up), 1 = East (right), 2 = South (down), 3 = West (left)
        direction = int((angle + math.pi / 4) / (math.pi / 2)) % 4
        return direction

    def mark_free_path_advanced(self, start_x, start_y, end_x, end_y):
        """Advanced ray tracing that stops at obstacles (like original Bresenham)"""
        cells = []
        dx = abs(end_x - start_x)
        dy = abs(end_y - start_y)
        x, y = start_x, start_y
        x_inc = 1 if end_x > start_x else -1
        y_inc = 1 if end_y > start_y else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        while True:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                cells.append((x, y))
                # Stop if we hit an obstacle (but not at robot position)
                if self.grid[y, x] == 1 and (x, y) != (start_x, start_y):
                    break

            if x == end_x and y == end_y:
                break

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        return cells

    def find_frontiers(self):
        """Find frontier cells (free cells next to unknown areas)"""
        frontiers = []

        for y in range(1, self.grid_size - 1):
            for x in range(1, self.grid_size - 1):
                # Must be free space
                if self.grid[y, x] != 0:
                    continue

                # Check if any neighbor is unknown
                has_unknown_neighbor = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if self.grid[y + dy, x + dx] == -1:
                        has_unknown_neighbor = True
                        break

                if has_unknown_neighbor:
                    frontiers.append((x, y))

        return frontiers

    def find_nearest_frontier(self):
        """Find the nearest safe frontier to explore"""
        if not self.robot_pos:
            return None

        frontiers = self.find_frontiers()
        if not frontiers:
            return None

        rx, ry = self.robot_pos
        best_frontier = None
        min_distance = float('inf')

        for fx, fy in frontiers:
            # Check if frontier is safe (not too close to obstacles)
            if not self.is_safe_position(fx, fy):
                continue

            # Calculate distance
            distance = math.sqrt((fx - rx) ** 2 + (fy - ry) ** 2)

            # Prefer frontiers that are not too close or too far
            if 3 < distance < 20 and distance < min_distance:
                min_distance = distance
                best_frontier = (fx, fy)

        return best_frontier

    def is_safe_position(self, x, y):
        """Check if a position is safe (away from obstacles)"""
        for dx in range(-self.safe_distance, self.safe_distance + 1):
            for dy in range(-self.safe_distance, self.safe_distance + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.grid[ny, nx] == 1:  # Obstacle nearby
                        return False
        return True

    def is_path_clear(self, target_x, target_y):
        """Check if path to target is clear"""
        if not self.robot_pos:
            return False

        rx, ry = self.robot_pos

        # Simple obstacle check in direction of target
        dx = 1 if target_x > rx else -1 if target_x < rx else 0
        dy = 1 if target_y > ry else -1 if target_y < ry else 0

        # Check next few cells in that direction
        for i in range(1, 4):
            check_x = rx + dx * i
            check_y = ry + dy * i

            if not (0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size):
                return False

            if self.grid[check_y, check_x] == 1:  # Obstacle
                return False

        return True

    def control_loop(self):
        """Main control logic - simplified state machine"""
        if not self.robot_pos:
            self.stop_robot()
            return

        # State machine
        if self.state == "EXPLORING":
            # Look for new frontier
            self.target = self.find_nearest_frontier()
            if self.target:
                self.state = "MOVING_TO_TARGET"
                self.get_logger().info(f"New target: {self.target}")

                # Publish goal for visualization
                goal_msg = Point()
                goal_msg.x = float(self.target[0])
                goal_msg.y = float(self.target[1])
                self.goal_pub.publish(goal_msg)
            else:
                # No frontiers found, just turn to look around
                self.turn_to_explore()

        elif self.state == "MOVING_TO_TARGET":
            if self.target is None:
                self.state = "EXPLORING"
                return

            # Check if we reached the target (or close enough)
            rx, ry = self.robot_pos
            tx, ty = self.target
            distance = math.sqrt((tx - rx) ** 2 + (ty - ry) ** 2)

            if distance < 2.0:  # Close enough
                self.get_logger().info("Target reached!")
                self.state = "EXPLORING"
                self.target = None
                self.stop_robot()
                return

            # Move toward target
            self.move_toward_target()

    def move_toward_target(self):
        """Move robot toward current target"""
        if not self.target or not self.robot_pos:
            return

        rx, ry = self.robot_pos
        tx, ty = self.target

        # Calculate desired heading
        target_angle = math.atan2(ty - ry, tx - rx)
        angle_diff = target_angle - self.robot_angle

        # Normalize angle difference
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # If we need to turn significantly, just turn
        if abs(angle_diff) > 0.3:  # ~17 degrees
            twist = Twist()
            twist.angular.z = self.angular_speed if angle_diff > 0 else -self.angular_speed
            self.cmd_pub.publish(twist)
            return

        # Check if path is clear
        if not self.is_path_clear(tx, ty):
            self.get_logger().warn("Path blocked, finding new target")
            self.state = "EXPLORING"
            self.target = None
            return

        # Move forward
        twist = Twist()
        twist.linear.x = self.linear_speed
        # Small angular correction
        twist.angular.z = angle_diff * 0.5
        self.cmd_pub.publish(twist)

    def turn_to_explore(self):
        """Turn to explore when no frontiers are found"""
        twist = Twist()
        twist.angular.z = self.angular_speed * 0.5  # Slow turn
        self.cmd_pub.publish(twist)

    def stop_robot(self):
        """Stop the robot"""
        twist = Twist()
        self.cmd_pub.publish(twist)

    def publish_map(self):
        """Publish the occupancy grid"""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.info.resolution = self.cell_size
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size
        msg.info.origin.position.x = -self.map_range
        msg.info.origin.position.y = -self.map_range

        # Convert grid to ROS format (0-100 scale)
        ros_grid = np.zeros_like(self.grid, dtype=np.int8)
        ros_grid[self.grid == 0] = 0  # Free space
        ros_grid[self.grid == 1] = 100  # Obstacles
        ros_grid[self.grid == -1] = -1  # Unknown

        msg.data = ros_grid.flatten().tolist()
        self.map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousExplorerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()