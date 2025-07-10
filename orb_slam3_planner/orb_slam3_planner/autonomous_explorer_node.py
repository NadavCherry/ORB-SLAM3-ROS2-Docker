#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import OccupancyGrid
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import math


class AutonomousExplorerNode(Node):
    def __init__(self):
        super().__init__('autonomous_explorer_node')

        # Keep your original parameters
        self.cell_size = 0.25  # meters per cell
        self.map_range = 20.0  # map extends ±25m
        self.grid_size = int(2 * self.map_range / self.cell_size)  # 100x100 grid

        # Robot state
        self.robot_pos = None  # (x, y) in grid coordinates
        self.robot_angle = 0.0  # current heading in radians
        self.current_pose = None

        # Enhanced grid with probability values
        self.occupancy_prob = np.full((self.grid_size, self.grid_size), 0.5, dtype=np.float32)
        self.update_count = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Probability thresholds
        self.occupied_threshold = 0.75  # Above this = occupied
        self.free_threshold = 0.35  # Below this = free

        # Navigation state
        self.target = None
        self.state = "EXPLORING"
        self.last_update = self.get_clock().now()

        # Collision avoidance
        self.collision_counter = 0
        self.stuck_counter = 0
        self.last_robot_pos = None

        # Enhanced mapping parameters
        self.min_points_for_obstacle = 20
        self.height_min = 0.1  # Minimum height for obstacles
        self.height_max = 2.0  # Maximum height for obstacles

        # Camera parameters
        self.camera_fov = math.radians(60)
        self.camera_range = 10.0  # Maximum reliable camera range in meters

        # Probability updates - more aggressive for walls
        self.obstacle_prob_increment = 0.2  # Stronger evidence for obstacles
        self.free_prob_decrement = -0.05  # Weaker free space updates
        self.freeze_update_count = 8

        # Movement parameters
        self.linear_speed = 0.6
        self.angular_speed = 0.5
        self.safe_distance = 4 # cells

        # Track exploration history
        self.visited_targets = set()  # Remember where we've been
        self.exploration_radius = 2  # How close counts as "visited"

        # Adaptive speed based on environment
        self.adaptive_speed = True
        self.min_linear_speed = 0.3
        self.max_linear_speed = 0.8

        # Better frontier scoring
        self.use_frontier_scoring = True
        self.last_frontier_update = self.get_clock().now()

        # ROS setup (keeping your exact publishers)
        self.create_subscription(PointCloud2, '/orb_slam3/landmarks_raw', self.pointcloud_callback, 10)
        self.create_subscription(PoseStamped, '/robot_pose_slam', self.pose_callback, 10)

        # Keep your original publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/planner_occupancy_grid', 10)
        self.goal_pub = self.create_publisher(Point, '/goal_grid_pos', 10)
        self.robot_pos_pub = self.create_publisher(Point, '/robot_grid_pos', 10)

        # Add direction visualization publisher
        self.direction_pub = self.create_publisher(PoseStamped, '/robot_direction', 10)

        # Main control timer
        self.create_timer(0.5, self.control_loop)  # 2Hz

        self.get_logger().info("Autonomous Explorer Node started!")
        self.get_logger().info(f"Grid size: {self.grid_size}x{self.grid_size}, Cell size: {self.cell_size}m")
        self.get_logger().info(f"Camera FOV: {math.degrees(self.camera_fov)}°, Range: {self.camera_range}m")

    def pose_callback(self, msg):
        """Update robot position from SLAM """
        self.current_pose = msg.pose

        # Convert to grid coordinates
        world_x = msg.pose.position.x
        world_y = msg.pose.position.y

        grid_x = int((world_x + self.map_range) / self.cell_size)
        grid_y = int((world_y + self.map_range) / self.cell_size)

        # Extract heading angle
        q = msg.pose.orientation
        self.robot_angle = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
            self.robot_pos = (grid_x, grid_y)

            # Publish robot position (keeping your exact format)
            robot_msg = Point()
            robot_msg.x = float(grid_x)
            robot_msg.y = float(grid_y)
            robot_msg.z = float(self.robot_angle)
            self.robot_pos_pub.publish(robot_msg)

    def publish_robot_direction(self):
        """Publish robot direction with FOV visualization"""
        if not self.current_pose:
            return

        dir_msg = PoseStamped()
        dir_msg.header.stamp = self.get_clock().now().to_msg()
        dir_msg.header.frame_id = "map"

        # Position
        dir_msg.pose.position = self.current_pose.position

        # Orientation (same as robot)
        dir_msg.pose.orientation = self.current_pose.orientation

        self.direction_pub.publish(dir_msg)

        # Log direction info periodically
        if hasattr(self, '_last_dir_log'):
            if (self.get_clock().now() - self._last_dir_log).nanoseconds > 5e9:  # Every 5 seconds
                self.get_logger().info(f"Robot angle: {math.degrees(self.robot_angle):.1f}°, "
                                       f"Direction: {self.get_robot_direction()}, "
                                       f"FOV: ±{math.degrees(self.camera_fov / 2):.1f}°")
                self._last_dir_log = self.get_clock().now()
        else:
            self._last_dir_log = self.get_clock().now()

    def normalize_angle(self, angle):
        """Normalize angle to range [-pi, pi]"""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def pointcloud_callback(self, msg):
        """Enhanced mapping with probabilistic updates"""
        if self.current_pose is None:
            return

        robot_world_x = self.current_pose.position.x
        robot_world_y = self.current_pose.position.y

        # # Clear area around robot (it must be free)
        # if self.robot_pos:
        #     self.clear_robot_area()

        # Process point cloud
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

        # Group points by cell for better obstacle detection
        cell_points = {}

        for x, y, z in points:
            # Filter by height
            if not (self.height_min <= z <= self.height_max):
                continue

            # Check if point is within camera FOV
            angle_to_point = math.atan2(y - robot_world_y, x - robot_world_x)
            angle_diff = angle_to_point - self.robot_angle
            angle_diff = self.normalize_angle(angle_diff)

            # Skip points outside FOV
            if abs(angle_diff) > self.camera_fov / 2:
                continue

            # Filter by range from robot
            dist_from_robot = math.sqrt((x - robot_world_x) ** 2 + (y - robot_world_y) ** 2)
            if dist_from_robot > self.camera_range:
                continue

            # Check if point is finite
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue

            # Convert to grid
            grid_x = int((x + self.map_range) / self.cell_size)
            grid_y = int((y + self.map_range) / self.cell_size)

            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                cell_key = (grid_x, grid_y)
                if cell_key not in cell_points:
                    cell_points[cell_key] = []
                cell_points[cell_key].append((x, y, z))

        # Update occupied cells with probability
        occupied_cells = set()
        for (gx, gy), points_in_cell in cell_points.items():
            if len(points_in_cell) >= self.min_points_for_obstacle:
                # Stronger evidence for obstacles with more points
                prob_increase = min(self.obstacle_prob_increment * len(points_in_cell), 0.5)
                self.update_cell_probability(gx, gy, prob_increase)
                occupied_cells.add((gx, gy))

                # Also mark neighboring cells for thicker walls
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            self.update_cell_probability(nx, ny, prob_increase * 0.5)

        # Ray trace to mark free cells (only within FOV)
        if self.robot_pos:
            self.update_free_space_probability(occupied_cells)

        # Decay probabilities slightly to handle dynamic changes
        self.decay_probabilities()

        # Publish map
        self.publish_map()

    def clear_robot_area(self):
        """Clear area around robot - it must be free"""
        if not self.robot_pos:
            return

        rx, ry = self.robot_pos
        clear_radius = 1

        for dx in range(-clear_radius, clear_radius + 1):
            for dy in range(-clear_radius, clear_radius + 1):
                nx, ny = rx + dx, ry + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    # Strong evidence that area around robot is free
                    self.occupancy_prob[ny, nx] = 0.1
                    self.update_count[ny, nx] += 1

    def update_cell_probability(self, x, y, prob_change):
        """Update cell probability using Bayesian-like update"""
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return

        # Bayesian update
        old_prob = self.occupancy_prob[y, x]

        if prob_change > 0:  # Evidence for occupied
            new_prob = old_prob + prob_change * (1 - old_prob)
        else:  # Evidence for free
            new_prob = old_prob + prob_change * old_prob

        self.occupancy_prob[y, x] = np.clip(new_prob, 0.01, 0.99)
        self.update_count[y, x] += 1

    def update_free_space_probability(self, occupied_cells):
        """Mark free space using ray tracing within camera FOV"""
        if not self.robot_pos:
            return

        robot_gx, robot_gy = self.robot_pos

        # First, only trace to visible obstacles within FOV
        for (gx, gy) in occupied_cells:
            # Check if obstacle is within camera FOV
            dx = gx - robot_gx
            dy = gy - robot_gy
            angle_to_obstacle = math.atan2(dy, dx)

            # Normalize angle difference
            angle_diff = angle_to_obstacle - self.robot_angle
            angle_diff = self.normalize_angle(angle_diff)

            # Only process if within FOV
            if abs(angle_diff) <= self.camera_fov / 2:
                cells_on_ray = self.bresenham_line(robot_gx, robot_gy, gx, gy)

                # Mark all cells except the obstacle as free
                for (x, y) in cells_on_ray[:-1]:
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        self.update_cell_probability(x, y, self.free_prob_decrement * 1.5)

        # Cast rays only within camera FOV
        fov_half = self.camera_fov / 2
        num_rays = int(self.camera_fov / math.radians(5))  # Ray every 5 degrees within FOV
        max_range_cells = int(self.camera_range / self.cell_size)

        for i in range(num_rays):
            # Calculate angle within FOV
            angle_offset = -fov_half + (i * self.camera_fov / (num_rays - 1))
            angle = self.robot_angle + angle_offset

            # Cast ray until we hit something or reach max range
            hit_obstacle = False
            for dist in range(1, max_range_cells):
                x = int(robot_gx + dist * math.cos(angle))
                y = int(robot_gy + dist * math.sin(angle))

                if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                    break

                # Check if we hit an obstacle with high confidence
                if self.occupancy_prob[y, x] > self.occupied_threshold and self.update_count[y, x] > 3:
                    hit_obstacle = True
                    break

                # Only mark as free if we haven't hit an obstacle yet
                if not hit_obstacle:
                    # Weaker update for radial rays
                    self.update_cell_probability(x, y, self.free_prob_decrement)

    def bresenham_line(self, start_x, start_y, end_x, end_y):
        """Bresenham's line algorithm """
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

            if x == end_x and y == end_y:
                break

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        return cells

    def decay_probabilities(self):
        """Slowly decay probabilities toward unknown to handle dynamic environments"""
        # Only decay cells that haven't been updated recently
        decay_factor = 0.99

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.update_count[y, x] < self.freeze_update_count:  # Low confidence cells
                    old_prob = self.occupancy_prob[y, x]
                    # Decay toward 0.5 (unknown)
                    self.occupancy_prob[y, x] = 0.5 + (old_prob - 0.5) * decay_factor

    def get_occupancy_value(self, x, y):
        """Convert probability to discrete occupancy value"""
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return -1  # Unknown

        prob = self.occupancy_prob[y, x]
        updates = self.update_count[y, x]

        # Need minimum updates to be confident
        if updates < 2:
            return -1  # Unknown

        if prob > self.occupied_threshold:
            return 1  # Occupied
        elif prob < self.free_threshold:
            return 0  # Free
        else:
            return -1  # Unknown

    def get_robot_direction(self):
        """Convert robot angle to discrete direction"""
        angle = self.robot_angle
        if angle < 0:
            angle += 2 * math.pi
        direction = int((angle + math.pi / 4) / (math.pi / 2)) % 4
        return direction

    def find_frontiers(self):
        """Find frontier cells using probability-based occupancy"""
        frontiers = []

        for y in range(1, self.grid_size - 1):
            for x in range(1, self.grid_size - 1):
                # Must be free space
                if self.get_occupancy_value(x, y) != 0:
                    continue

                # Check if any neighbor is unknown
                has_unknown_neighbor = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if self.get_occupancy_value(x + dx, y + dy) == -1:
                        has_unknown_neighbor = True
                        break

                if has_unknown_neighbor:
                    frontiers.append((x, y))

        return frontiers

    def calculate_frontier_score(self, fx, fy, rx, ry):
        """Calculate score for a frontier based on multiple factors"""
        # Distance to frontier
        distance = math.sqrt((fx - rx) ** 2 + (fy - ry) ** 2)

        # Check if we've been near this frontier before
        novelty_bonus = 1.0
        for visited_x, visited_y in self.visited_targets:
            dist_to_visited = math.sqrt((fx - visited_x) ** 2 + (fy - visited_y) ** 2)
            if dist_to_visited < self.exploration_radius:
                novelty_bonus = 0.3  # Reduce score for previously visited areas
                break

        # Count unknown cells around the frontier (information gain)
        unknown_count = 0
        check_radius = 3
        for dx in range(-check_radius, check_radius + 1):
            for dy in range(-check_radius, check_radius + 1):
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.get_occupancy_value(nx, ny) == -1:
                        unknown_count += 1

        # Calculate angle to frontier (prefer frontiers in current direction)
        angle_to_frontier = math.atan2(fy - ry, fx - rx)
        angle_diff = abs(self.normalize_angle(angle_to_frontier - self.robot_angle))
        angle_factor = 1.0 - (angle_diff / math.pi) * 0.3  # Small penalty for turning

        # Combine factors into score (lower is better)
        # Distance is primary factor, but modified by others
        score = distance / novelty_bonus / (1 + unknown_count * 0.1) / angle_factor

        return score

    def find_best_frontier(self):
        """Find the best frontier using scoring system"""
        if not self.robot_pos:
            return None

        frontiers = self.find_frontiers()
        if not frontiers:
            return None

        rx, ry = self.robot_pos
        best_frontier = None
        best_score = float('inf')

        for fx, fy in frontiers:
            # Check if frontier is safe
            if not self.is_safe_position(fx, fy):
                continue

            # Skip if too close or too far
            distance = math.sqrt((fx - rx) ** 2 + (fy - ry) ** 2)
            if distance < 3 or distance > 30:
                continue

            # Calculate score
            score = self.calculate_frontier_score(fx, fy, rx, ry)

            if score < best_score:
                best_score = score
                best_frontier = (fx, fy)

        return best_frontier

    def find_nearest_frontier(self):
        """Wrapper to use either scoring or simple nearest frontier"""
        if self.use_frontier_scoring:
            return self.find_best_frontier()
        else:
            # Original implementation
            if not self.robot_pos:
                return None

            frontiers = self.find_frontiers()
            if not frontiers:
                return None

            rx, ry = self.robot_pos
            best_frontier = None
            min_distance = float('inf')

            for fx, fy in frontiers:
                # Check if frontier is safe
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
        """Check if position is safe using probability map"""
        for dx in range(-self.safe_distance, self.safe_distance + 1):
            for dy in range(-self.safe_distance, self.safe_distance + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.occupancy_prob[ny, nx] > self.occupied_threshold:
                        return False
        return True

    def check_collision_ahead(self):
        """Check for obstacles ahead in a cone matching robot width"""
        if not self.robot_pos:
            return False

        rx, ry = self.robot_pos

        # Robot parameters
        robot_width_cells = 1  # How wide the robot is in cells
        check_distance = self.safe_distance  # How far ahead to check

        # Check in a cone/arc in front of the robot
        for dist in range(1, check_distance + 1):
            # Width of check area increases slightly with distance
            width_at_dist = max(1, robot_width_cells - dist // 3)

            # Check cells in an arc at this distance
            for offset in range(-width_at_dist, width_at_dist + 1):
                # Calculate angle for this offset
                # Offset perpendicular to robot direction
                check_angle = self.robot_angle + (offset * 0.2 / dist)  # Narrower cone

                check_x = int(rx + dist * math.cos(check_angle))
                check_y = int(ry + dist * math.sin(check_angle))

                if 0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size:
                    if self.occupancy_prob[check_y, check_x] > self.occupied_threshold:
                        return True

        return False

    def calculate_obstacle_density(self):
        """Calculate obstacle density around robot for adaptive speed"""
        if not self.robot_pos:
            return 0.0

        rx, ry = self.robot_pos
        obstacle_count = 0
        check_radius = 5
        total_cells = 0

        for dx in range(-check_radius, check_radius + 1):
            for dy in range(-check_radius, check_radius + 1):
                nx, ny = rx + dx, ry + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    total_cells += 1
                    if self.occupancy_prob[ny, nx] > self.occupied_threshold:
                        obstacle_count += 1

        return obstacle_count / total_cells if total_cells > 0 else 0.0

    def get_adaptive_speed(self):
        """Calculate adaptive speed based on environment"""
        if not self.adaptive_speed:
            return self.linear_speed

        # Get obstacle density
        density = self.calculate_obstacle_density()

        # Calculate speed (inverse relationship with density)
        # High density = slow speed, low density = fast speed
        speed_range = self.max_linear_speed - self.min_linear_speed
        adaptive_speed = self.max_linear_speed - (density * speed_range)

        return np.clip(adaptive_speed, self.min_linear_speed, self.max_linear_speed)

    def is_stuck(self):
        """Check if robot is stuck"""
        if not self.robot_pos or not self.last_robot_pos:
            return False

        # Check if position hasn't changed
        if self.robot_pos == self.last_robot_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        return self.stuck_counter > 10  # Stuck for 5 seconds

    def control_loop(self):
        """Enhanced control logic with collision avoidance (safe forward motion only)"""
        if not self.robot_pos:
            self.stop_robot()
            return

        # Check if stuck
        if self.is_stuck():
            self.get_logger().warn("Robot is stuck! Executing recovery...")
            self.state = "RECOVERY"
            self.stuck_counter = 0

        # Update last position
        self.last_robot_pos = self.robot_pos

        # State machine
        if self.state == "COLLISION_AVOIDANCE":
            # Turn right to avoid collision
            if self.collision_counter > 0:
                twist = Twist()
                twist.angular.z = -self.angular_speed  # Turn right
                self.cmd_pub.publish(twist)
                self.collision_counter -= 1
            else:
                self.state = "EXPLORING"

        elif self.state == "RECOVERY":
            # Back up and turn
            twist = Twist()
            twist.linear.x = -self.linear_speed * 0.8
            twist.angular.z = -self.angular_speed * 1.5
            self.cmd_pub.publish(twist)
            self.state = "EXPLORING"

        elif self.state == "EXPLORING":
            # Look for new frontier
            self.target = self.find_nearest_frontier()
            if self.target:
                self.state = "MOVING_TO_TARGET"
                self.get_logger().info(f"New target: {self.target}")

                # Publish goal for visualization
                goal_msg = Point()
                goal_msg.x = float(self.target[0])
                goal_msg.y = float(self.target[1])
                goal_msg.z = 0.0
                self.goal_pub.publish(goal_msg)
            else:
                # No frontiers found, just turn to explore
                self.turn_to_explore()

        elif self.state == "MOVING_TO_TARGET":
            if self.target is None:
                self.state = "EXPLORING"
                return

            # Check if we reached the target
            rx, ry = self.robot_pos
            tx, ty = self.target
            distance = math.sqrt((tx - rx) ** 2 + (ty - ry) ** 2)

            if distance < 2.0:  # Close enough
                self.get_logger().info("Target reached!")
                self.visited_targets.add((tx, ty))
                self.state = "EXPLORING"
                self.target = None
                self.stop_robot()
                return

            # Check collision ahead *only before moving forward*
            if self.check_collision_ahead():
                self.get_logger().warn("Obstacle ahead! Halting forward motion.")
                self.stop_robot()
                return

            # Move toward target
            self.move_toward_target()

    def move_toward_target(self):
        """Move robot toward current target with enhanced obstacle checking"""
        if not self.target or not self.robot_pos:
            return

        rx, ry = self.robot_pos
        tx, ty = self.target

        # Calculate desired heading
        target_angle = math.atan2(ty - ry, tx - rx)
        angle_diff = target_angle - self.robot_angle

        # Normalize angle difference
        angle_diff = self.normalize_angle(angle_diff)

        # Get adaptive speed
        current_speed = self.get_adaptive_speed()

        # If we need to turn significantly
        if abs(angle_diff) > 0.3:  # ~17 degrees
            twist = Twist()
            twist.angular.z = self.angular_speed if angle_diff > 0 else -self.angular_speed
            self.cmd_pub.publish(twist)
            return

        # Move forward with angular correction
        twist = Twist()
        twist.linear.x = current_speed
        twist.angular.z = angle_diff * 0.5
        self.cmd_pub.publish(twist)

    def turn_to_explore(self):
        """Turn to explore when no frontiers are found"""
        # Systematic exploration pattern
        twist = Twist()
        turn_direction = 1 if int(self.get_clock().now().nanoseconds / 1e9) % 10 < 5 else -1
        twist.angular.z = self.angular_speed * 0.5 * turn_direction
        twist.linear.x = self.linear_speed * 0.3  # Slow forward movement while turning
        self.cmd_pub.publish(twist)

    def stop_robot(self):
        """Stop the robot"""
        twist = Twist()
        self.cmd_pub.publish(twist)

    def publish_map(self):
        """Publish the occupancy grid with probability-based values"""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.info.resolution = self.cell_size
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size
        msg.info.origin.position.x = -self.map_range
        msg.info.origin.position.y = -self.map_range

        # Convert probability grid to ROS format
        ros_grid = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                value = self.get_occupancy_value(x, y)
                if value == -1:
                    ros_grid.append(-1)  # Unknown
                elif value == 0:
                    ros_grid.append(0)  # Free
                else:
                    ros_grid.append(100)  # Occupied

        msg.data = ros_grid
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