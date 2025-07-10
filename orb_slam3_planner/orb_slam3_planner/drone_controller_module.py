import math
import numpy as np
from geometry_msgs.msg import Twist

class DroneController:
    """
    DroneController is responsible for commanding the robot's movement,
    handling obstacle avoidance, stuck recovery, and adaptive speed control.
    """

    def __init__(self, node):
        """
        Initialize the DroneController with references to the shared main node.

        Args:
            node (rclpy.node.Node): The central ROS2 node containing shared state and publishers.
        """
        self.node = node
        self.get_logger = node.get_logger
        self.get_clock = node.get_clock
        self.cmd_pub = node.cmd_pub  # Publisher to /cmd_vel
        self.get_logger().info("✅ DroneController initialized")

    def move_toward_target(self):
        """
        Move the robot towards the currently assigned target while maintaining heading alignment.
        If significant angular correction is needed, the robot will rotate in place before moving forward.
        """
        if not self.node.target or not self.node.robot_pos:
            return

        rx, ry = self.node.robot_pos
        tx, ty = self.node.target

        target_angle = math.atan2(ty - ry, tx - rx)
        angle_diff = self.node.normalize_angle(target_angle - self.node.robot_angle)

        speed = self.get_adaptive_speed()

        if abs(angle_diff) > 0.3:
            twist = Twist()
            twist.angular.z = self.node.angular_speed if angle_diff > 0 else -self.node.angular_speed
            self.cmd_pub.publish(twist)
            return

        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = angle_diff * 0.5
        self.cmd_pub.publish(twist)

    def stop_robot(self):
        """
        Immediately stop all robot motion by publishing zero velocities.
        """
        twist = Twist()
        self.cmd_pub.publish(twist)

    def turn_to_explore(self):
        """
        Rotate in place or gently move forward to facilitate exploration when no frontiers are available.
        """
        twist = Twist()
        turn_direction = 1 if int(self.get_clock().now().nanoseconds / 1e9) % 10 < 5 else -1
        twist.angular.z = self.node.angular_speed * 0.5 * turn_direction
        twist.linear.x = self.node.linear_speed * 0.3
        self.cmd_pub.publish(twist)

    def is_stuck(self):
        """
        Check whether the robot is stuck by comparing current position with the last known position.

        Returns:
            bool: True if the robot is not making progress; otherwise, False.
        """
        if not self.node.robot_pos or not self.node.last_robot_pos:
            return False

        if self.node.robot_pos == self.node.last_robot_pos:
            self.node.stuck_counter += 1
        else:
            self.node.stuck_counter = 0

        return self.node.stuck_counter > 10

    def check_collision_ahead(self):
        """
        Look ahead of the robot's current heading for obstacles using the occupancy grid.
        A conical detection area is used to check for potential collisions.

        Returns:
            bool: True if an obstacle is detected ahead; False otherwise.
        """
        if not self.node.robot_pos:
            return False

        rx, ry = self.node.robot_pos
        robot_width_cells = 1
        check_distance = self.node.safe_distance

        for dist in range(1, check_distance + 1):
            width_at_dist = max(1, robot_width_cells - dist // 3)

            for offset in range(-width_at_dist, width_at_dist + 1):
                check_angle = self.node.robot_angle + (offset * 0.2 / dist)
                check_x = int(rx + dist * math.cos(check_angle))
                check_y = int(ry + dist * math.sin(check_angle))

                if 0 <= check_x < self.node.grid_size and 0 <= check_y < self.node.grid_size:
                    if self.node.occupancy_prob[check_y, check_x] > self.node.occupied_threshold:
                        return True

        return False

    def calculate_obstacle_density(self):
        """
        Calculate the local density of obstacles around the robot within a fixed radius.

        Returns:
            float: Ratio of occupied cells to total cells within the check area.
        """
        if not self.node.robot_pos:
            return 0.0

        rx, ry = self.node.robot_pos
        obstacle_count = 0
        check_radius = 5
        total_cells = 0

        for dx in range(-check_radius, check_radius + 1):
            for dy in range(-check_radius, check_radius + 1):
                nx, ny = rx + dx, ry + dy
                if 0 <= nx < self.node.grid_size and 0 <= ny < self.node.grid_size:
                    total_cells += 1
                    if self.node.occupancy_prob[ny, nx] > self.node.occupied_threshold:
                        obstacle_count += 1

        return obstacle_count / total_cells if total_cells > 0 else 0.0

    def get_adaptive_speed(self):
        """
        Compute the robot's linear speed dynamically based on local obstacle density.

        Returns:
            float: The selected linear speed (m/s) within allowed speed bounds.
        """
        if not self.node.adaptive_speed:
            return self.node.linear_speed

        density = self.calculate_obstacle_density()
        speed_range = self.node.max_linear_speed - self.node.min_linear_speed
        adaptive_speed = self.node.max_linear_speed - (density * speed_range)

        return np.clip(adaptive_speed, self.node.min_linear_speed, self.node.max_linear_speed)
