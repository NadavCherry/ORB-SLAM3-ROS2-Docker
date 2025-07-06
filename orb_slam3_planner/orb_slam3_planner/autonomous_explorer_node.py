#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from tf2_ros import TransformListener, Buffer
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import threading
import cv2
from scipy import ndimage
import math
import os
from typing import Optional, Tuple, List, Set


class AutonomousExplorerNode(Node):
    def __init__(self):
        super().__init__('autonomous_explorer_node')

        # Grid parameters (same as enhanced map builder)
        self.resolution = 0.1  # meters per pixel
        self.map_size = 50.0  # meters
        self.grid_size = int(self.map_size / self.resolution)

        # Planner grid parameters
        self.planner_cell_size = 0.5  # Each planner cell is 0.5m x 0.5m
        self.planner_grid_resolution = int(self.planner_cell_size / self.resolution)

        # Initialize grids
        self.raw_grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        self.planner_grid = None
        self.safe_grid = None  # Grid with safety margins

        # Robot state
        self.current_pose = None
        self.robot_grid_pos = None  # (x, y) in planner grid
        self.robot_facing = 'NORTH'  # Current facing direction
        self.pose_lock = threading.Lock()

        # Planning state
        self.current_goal = None  # Target frontier cell
        self.current_action = 'STAY'
        self.action_start_time = None
        self.action_duration = 1.0  # seconds per action
        self.last_action = 'STAY'
        self.stuck_counter = 0

        # Frontier detection
        self.frontiers = set()

        # Movement parameters
        self.linear_speed = 0.15  # m/s (reduced for safety)
        self.angular_speed = 0.4  # rad/s (reduced for safety)
        self.goal_reached_distance = 1  # cells - don't need to reach exactly
        self.min_frontier_distance = 0.5  # cells - minimum distance to select a frontier

        # Safety parameters
        self.obstacle_inflate_radius = 1  # cells to inflate obstacles
        self.emergency_stop_distance = 0.2  # cells

        # Map building parameters
        self.frame_count = 0
        self.rebuild_interval = 20
        self.z_min = 0.1
        self.z_max = 2.0
        self.max_range = 20.0
        self.min_points_for_obstacle = 5

        # ROS setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(PointCloud2, '/orb_slam3/landmarks_raw', self.cloud_callback, 10)
        self.create_subscription(PoseStamped, '/robot_pose_slam', self.pose_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/planner_occupancy_grid', 10)

        # Main control timer (runs at 10Hz)
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Planning timer (runs at 1Hz)
        self.planning_timer = self.create_timer(1.0, self.planning_loop)

        # Visualization
        self.save_visualizations = True
        self.viz_path = '/tmp/autonomous_explorer'
        os.makedirs(self.viz_path, exist_ok=True)

        self.get_logger().info("Autonomous Explorer Node started")
        self.get_logger().info(f"Planner grid cell size: {self.planner_cell_size}m")
        self.get_logger().info(f"Goal reached distance: {self.goal_reached_distance} cells")
        self.get_logger().info(f"Safety margin: {self.obstacle_inflate_radius} cells")

    def pose_callback(self, msg):
        """Update robot pose from SLAM"""
        with self.pose_lock:
            self.current_pose = msg.pose

            # Calculate facing direction from quaternion
            # Simple approach: use yaw angle
            q = msg.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))

            # Convert yaw to discrete direction
            # Normalize to [0, 2π]
            if yaw < 0:
                yaw += 2 * math.pi

            # Convert to 8 directions (0°, 45°, 90°, etc.)
            direction_index = int((yaw + math.pi / 8) / (math.pi / 4)) % 8
            directions = ['EAST', 'NORTHEAST', 'NORTH', 'NORTHWEST',
                          'WEST', 'SOUTHWEST', 'SOUTH', 'SOUTHEAST']

            # For now, simplify to 4 cardinal directions
            if direction_index in [0, 1]:
                self.robot_facing = 'EAST'
            elif direction_index in [2, 3]:
                self.robot_facing = 'NORTH'
            elif direction_index in [4, 5]:
                self.robot_facing = 'WEST'
            else:
                self.robot_facing = 'SOUTH'

    def cloud_callback(self, msg):
        """Process point cloud and update map (same as enhanced map builder)"""
        with self.pose_lock:
            if self.current_pose is None:
                return
            robot_x = self.current_pose.position.x
            robot_y = self.current_pose.position.y

        self.frame_count += 1

        # Periodically clear grid to stay in sync with SLAM
        if self.frame_count % self.rebuild_interval == 0:
            self.raw_grid.fill(-1)

        # Process point cloud (same logic as before)
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

        # First pass: collect all occupied cells
        occupied_cells = set()
        valid_points = []

        for x, y, z in points:
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue
            if z < self.z_min or z > self.z_max:
                continue
            distance = np.sqrt((x - robot_x) ** 2 + (y - robot_y) ** 2)
            if distance > self.max_range:
                continue

            gx = int((x + self.map_size / 2) / self.resolution)
            gy = int((y + self.map_size / 2) / self.resolution)

            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                occupied_cells.add((gx, gy))
                valid_points.append((gx, gy))

        # Mark occupied cells first
        for (gx, gy) in occupied_cells:
            self.raw_grid[gy, gx] = 100

        # Second pass: ray trace to mark free cells
        robot_gx = int((robot_x + self.map_size / 2) / self.resolution)
        robot_gy = int((robot_y + self.map_size / 2) / self.resolution)

        observed_cells = set()

        for (gx, gy) in valid_points:
            cells_on_ray = self.bresenham_line_with_obstacles(
                robot_gx, robot_gy, gx, gy, self.raw_grid
            )
            observed_cells.update(cells_on_ray[:-1])

        # Mark free cells
        for (gx, gy) in observed_cells:
            if self.raw_grid[gy, gx] == -1:
                self.raw_grid[gy, gx] = 0

        # Process into planner grid
        self.process_planner_grid()

        # Update frontiers
        self.update_frontiers()

        # Publish map
        self.publish_planner_grid()

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
        planner_width = self.grid_size // self.planner_grid_resolution
        planner_height = self.grid_size // self.planner_grid_resolution

        self.planner_grid = np.full((planner_height, planner_width), -1, dtype=np.int8)

        for py in range(planner_height):
            for px in range(planner_width):
                start_x = px * self.planner_grid_resolution
                end_x = min((px + 1) * self.planner_grid_resolution, self.grid_size)
                start_y = py * self.planner_grid_resolution
                end_y = min((py + 1) * self.planner_grid_resolution, self.grid_size)

                sub_region = self.raw_grid[start_y:end_y, start_x:end_x]

                occupied_count = np.sum(sub_region == 100)
                free_count = np.sum(sub_region == 0)
                unknown_count = np.sum(sub_region == -1)
                total_cells = sub_region.size

                if occupied_count >= self.min_points_for_obstacle:
                    self.planner_grid[py, px] = 100
                elif free_count > total_cells * 0.5:
                    self.planner_grid[py, px] = 0
                elif unknown_count > total_cells * 0.8:
                    self.planner_grid[py, px] = -1
                else:
                    self.planner_grid[py, px] = 0

        # Create safety grid with inflated obstacles
        self.create_safety_grid()

        # Update robot position in planner grid
        with self.pose_lock:
            if self.current_pose is not None:
                robot_x = self.current_pose.position.x
                robot_y = self.current_pose.position.y

                pgx = int((robot_x + self.map_size / 2) / self.planner_cell_size)
                pgy = int((robot_y + self.map_size / 2) / self.planner_cell_size)

                if 0 <= pgx < planner_width and 0 <= pgy < planner_height:
                    self.robot_grid_pos = (pgx, pgy)
                    self.planner_grid[pgy, pgx] = 0
                    self.safe_grid[pgy, pgx] = 0

    def create_safety_grid(self):
        """Create a grid with inflated obstacles for safe navigation"""
        if self.planner_grid is None:
            return

        self.safe_grid = self.planner_grid.copy()
        height, width = self.planner_grid.shape

        # Find all obstacle cells
        obstacle_cells = []
        for y in range(height):
            for x in range(width):
                if self.planner_grid[y, x] == 100:
                    obstacle_cells.append((x, y))

        # Inflate obstacles
        for ox, oy in obstacle_cells:
            for dy in range(-self.obstacle_inflate_radius, self.obstacle_inflate_radius + 1):
                for dx in range(-self.obstacle_inflate_radius, self.obstacle_inflate_radius + 1):
                    nx, ny = ox + dx, oy + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        # Don't overwrite actual obstacles or unknown cells
                        if self.safe_grid[ny, nx] == 0:
                            self.safe_grid[ny, nx] = 50  # Mark as dangerous (near obstacle)

    def update_frontiers(self):
        """Find frontier cells (known free cells bordering unknown cells)"""
        if self.planner_grid is None:
            return

        self.frontiers.clear()
        height, width = self.planner_grid.shape

        for y in range(height):
            for x in range(width):
                # Skip if not free
                if self.planner_grid[y, x] != 0:
                    continue

                # Skip robot's current position
                if self.robot_grid_pos and (x, y) == self.robot_grid_pos:
                    continue

                # Check if any neighbor is unknown
                is_frontier = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if self.planner_grid[ny, nx] == -1:
                            is_frontier = True
                            break

                if is_frontier:
                    self.frontiers.add((x, y))

    def planning_loop(self):
        """Main planning loop - runs every second"""
        if self.planner_grid is None or self.robot_grid_pos is None:
            return

        # Log frontier count for debugging
        self.get_logger().info(f"Found {len(self.frontiers)} frontiers")

        # Save visualization for debugging
        if self.save_visualizations and self.frame_count % 5 == 0:
            self.save_debug_visualization()

        # Check if we need a new goal
        if self.current_goal is None or self.current_goal not in self.frontiers:
            # Find nearest frontier
            self.current_goal = self.find_nearest_frontier()
            if self.current_goal:
                self.get_logger().info(f"New goal selected: {self.current_goal}")

        if self.current_goal:
            # Determine action to move toward goal
            self.current_action = self.plan_action_to_goal()
            self.action_start_time = self.get_clock().now()

            self.get_logger().info(
                f"Planning: Robot at {self.robot_grid_pos} facing {self.robot_facing}, "
                f"Goal: {self.current_goal}, Action: {self.current_action}"
            )
        else:
            self.current_action = 'STAY'
            self.get_logger().info("No suitable frontiers found - all frontiers too close or none exist!")

    def find_nearest_frontier(self):
        """Find the nearest frontier cell to the robot"""
        if not self.frontiers or not self.robot_grid_pos:
            return None

        min_dist = float('inf')
        nearest = None
        rx, ry = self.robot_grid_pos

        # Find all suitable frontiers
        suitable_frontiers = []

        for fx, fy in self.frontiers:
            # Manhattan distance
            dist = abs(fx - rx) + abs(fy - ry)

            # Skip frontiers that are too close
            if dist < self.min_frontier_distance:
                continue

            # Skip frontiers in dangerous areas
            if self.safe_grid is not None and self.safe_grid[fy, fx] != 0:
                continue

            suitable_frontiers.append(((fx, fy), dist))

        # Sort by distance and select the nearest
        if suitable_frontiers:
            suitable_frontiers.sort(key=lambda x: x[1])
            nearest = suitable_frontiers[0][0]
            self.get_logger().info(f"Selected frontier at {nearest}, distance: {suitable_frontiers[0][1]}")
        else:
            self.get_logger().info("No suitable frontiers found")

        return nearest

    def plan_action_to_goal(self):
        """Plan next action to move toward goal with safety checks"""
        if not self.current_goal or not self.robot_grid_pos:
            return 'STAY'

        rx, ry = self.robot_grid_pos
        gx, gy = self.current_goal

        # Calculate distance to goal
        dist = abs(rx - gx) + abs(ry - gy)

        # If we're close enough to observe the frontier, consider goal reached
        if dist <= self.goal_reached_distance:
            self.get_logger().info(f"Close enough to observe frontier at {self.current_goal} (distance: {dist})")
            self.current_goal = None
            self.stuck_counter = 0
            return 'STAY'

        # Check if we're stuck
        if self.current_action == self.last_action and self.current_action != 'STAY':
            self.stuck_counter += 1
            if self.stuck_counter > 5:
                self.get_logger().warn("Robot appears stuck, trying different action")
                self.stuck_counter = 0
                return 'TURN_LEFT' if self.last_action == 'TURN_RIGHT' else 'TURN_RIGHT'
        else:
            self.stuck_counter = 0

        # Calculate direction to goal
        dx = gx - rx
        dy = gy - ry

        # Determine desired direction based on largest component
        if abs(dx) > abs(dy):
            desired_dir = 'EAST' if dx > 0 else 'WEST'
        else:
            desired_dir = 'NORTH' if dy > 0 else 'SOUTH'

        # Check if we're already facing the right direction
        if self.robot_facing == desired_dir:
            # Check if moving forward is safe
            if self.check_forward_safety():
                next_x, next_y = self.get_next_position(rx, ry, self.robot_facing)
                if self.is_valid_move(next_x, next_y):
                    self.last_action = self.current_action
                    return 'FORWARD'

            # Can't move forward, try to find alternative
            # Try turning to go around obstacle
            left_dir = self.get_left_direction(self.robot_facing)
            right_dir = self.get_right_direction(self.robot_facing)

            # Check which way is better
            left_x, left_y = self.get_direction_delta(left_dir)
            right_x, right_y = self.get_direction_delta(right_dir)

            left_clear = self.is_valid_move(rx + left_x, ry + left_y)
            right_clear = self.is_valid_move(rx + right_x, ry + right_y)

            if left_clear and not right_clear:
                self.last_action = self.current_action
                return 'TURN_LEFT'
            elif right_clear and not left_clear:
                self.last_action = self.current_action
                return 'TURN_RIGHT'
            else:
                # Both or neither clear, choose based on goal direction
                self.last_action = self.current_action
                return 'TURN_RIGHT'
        else:
            # Need to turn toward goal
            self.last_action = self.current_action
            return self.get_turn_direction(self.robot_facing, desired_dir)

    def get_left_direction(self, facing):
        """Get the direction to the left of current facing"""
        directions = ['NORTH', 'WEST', 'SOUTH', 'EAST']
        idx = directions.index(facing)
        return directions[(idx + 1) % 4]

    def get_right_direction(self, facing):
        """Get the direction to the right of current facing"""
        directions = ['NORTH', 'EAST', 'SOUTH', 'WEST']
        idx = directions.index(facing)
        return directions[(idx + 1) % 4]

    def get_direction_delta(self, direction):
        """Get the (dx, dy) for a given direction"""
        deltas = {
            'NORTH': (0, 1),
            'SOUTH': (0, -1),
            'EAST': (1, 0),
            'WEST': (-1, 0)
        }
        return deltas.get(direction, (0, 0))

    def get_next_position(self, x, y, facing):
        """Get next grid position given current position and facing"""
        if facing == 'NORTH':
            return x, y + 1
        elif facing == 'SOUTH':
            return x, y - 1
        elif facing == 'EAST':
            return x + 1, y
        elif facing == 'WEST':
            return x - 1, y
        return x, y

    def is_valid_move(self, x, y):
        """Check if a grid position is valid and safe to move to"""
        if self.safe_grid is None:
            return False

        height, width = self.safe_grid.shape
        if not (0 <= x < width and 0 <= y < height):
            return False

        # Can only move to free cells (not obstacles or danger zones)
        return self.safe_grid[y, x] == 0

    def check_forward_safety(self):
        """Check if moving forward is safe"""
        if not self.robot_grid_pos or self.safe_grid is None:
            return False

        rx, ry = self.robot_grid_pos

        # Check multiple cells ahead for safety
        for distance in range(1, self.emergency_stop_distance + 1):
            if self.robot_facing == 'NORTH':
                check_x, check_y = rx, ry + distance
            elif self.robot_facing == 'SOUTH':
                check_x, check_y = rx, ry - distance
            elif self.robot_facing == 'EAST':
                check_x, check_y = rx + distance, ry
            elif self.robot_facing == 'WEST':
                check_x, check_y = rx - distance, ry
            else:
                return False

            # Check bounds
            if not (0 <= check_x < self.safe_grid.shape[1] and
                    0 <= check_y < self.safe_grid.shape[0]):
                return False

            # Check for obstacles or danger zones
            if self.safe_grid[check_y, check_x] != 0:
                self.get_logger().warn(f"Obstacle detected {distance} cells ahead!")
                return False

        return True

    def get_turn_direction(self, current, desired):
        """Determine turn direction to face desired direction"""
        directions = ['NORTH', 'EAST', 'SOUTH', 'WEST']
        current_idx = directions.index(current)
        desired_idx = directions.index(desired)

        # Calculate shortest turn
        diff = (desired_idx - current_idx) % 4
        if diff == 1:
            return 'TURN_RIGHT'
        elif diff == 3:
            return 'TURN_LEFT'
        elif diff == 2:
            # 180 degree turn, choose either direction
            return 'TURN_RIGHT'
        else:
            return 'STAY'

    def control_loop(self):
        """Main control loop - executes movement commands with safety checks"""
        # Emergency stop if obstacle detected while moving forward
        if self.current_action == 'FORWARD' and not self.check_forward_safety():
            self.get_logger().warn("Emergency stop - obstacle detected!")
            self.stop_robot()
            self.current_action = 'STAY'
            return

        if self.current_action == 'STAY':
            self.stop_robot()
        elif self.current_action == 'FORWARD':
            self.move_forward()
        elif self.current_action == 'TURN_LEFT':
            self.turn_left()
        elif self.current_action == 'TURN_RIGHT':
            self.turn_right()

    def stop_robot(self):
        """Send stop command"""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

    def move_forward(self):
        """Send forward movement command with reduced speed near obstacles"""
        twist = Twist()

        # Check distance to nearest obstacle and reduce speed if close
        if self.robot_grid_pos and self.safe_grid is not None:
            # Simple speed reduction based on nearby obstacles
            speed = self.linear_speed

            # Check cells ahead
            rx, ry = self.robot_grid_pos
            next_x, next_y = self.get_next_position(rx, ry, self.robot_facing)

            if 0 <= next_x < self.safe_grid.shape[1] and 0 <= next_y < self.safe_grid.shape[0]:
                if self.safe_grid[next_y, next_x] == 50:  # Near obstacle
                    speed *= 0.5  # Reduce speed by half

        else:
            speed = self.linear_speed

        twist.linear.x = speed
        self.cmd_vel_pub.publish(twist)

    def turn_left(self):
        """Send left turn command"""
        twist = Twist()
        twist.angular.z = self.angular_speed
        self.cmd_vel_pub.publish(twist)

    def turn_right(self):
        """Send right turn command"""
        twist = Twist()
        twist.angular.z = -self.angular_speed
        self.cmd_vel_pub.publish(twist)

    def publish_planner_grid(self):
        """Publish the planner grid as OccupancyGrid message"""
        if self.planner_grid is None:
            return

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.resolution = self.planner_cell_size
        msg.info.width = self.planner_grid.shape[1]
        msg.info.height = self.planner_grid.shape[0]
        msg.info.origin.position.x = -self.map_size / 2
        msg.info.origin.position.y = -self.map_size / 2
        msg.data = self.planner_grid.flatten().tolist()

        self.map_pub.publish(msg)

    def save_debug_visualization(self):
        """Save a visualization of the current state for debugging"""
        if self.planner_grid is None:
            return

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))

        # Create visualization
        vis = np.zeros((*self.planner_grid.shape, 3))
        vis[self.planner_grid == 100] = [1, 0, 0]  # Red for obstacles
        vis[self.planner_grid == 0] = [0, 1, 0]  # Green for free
        vis[self.planner_grid == -1] = [0.5, 0.5, 0.5]  # Gray for unknown

        # Mark frontiers
        for fx, fy in self.frontiers:
            vis[fy, fx] = [0, 0, 1]  # Blue for frontiers

        # Mark robot position
        if self.robot_grid_pos:
            rx, ry = self.robot_grid_pos
            if 0 <= rx < self.planner_grid.shape[1] and 0 <= ry < self.planner_grid.shape[0]:
                vis[ry, rx] = [1, 1, 0]  # Yellow for robot

        # Mark goal
        if self.current_goal:
            gx, gy = self.current_goal
            if 0 <= gx < self.planner_grid.shape[1] and 0 <= gy < self.planner_grid.shape[0]:
                vis[gy, gx] = [1, 0, 1]  # Magenta for goal

        ax.imshow(vis, origin='lower')
        ax.set_title(f'Frame {self.frame_count} - Robot: {self.robot_grid_pos}, Goal: {self.current_goal}')

        # Add grid
        for i in range(self.planner_grid.shape[1] + 1):
            ax.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
        for i in range(self.planner_grid.shape[0] + 1):
            ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)

        # Save
        filename = os.path.join(self.viz_path, f'debug_frame_{self.frame_count:06d}.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()

        self.get_logger().info(f"Saved debug visualization to {filename}")


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