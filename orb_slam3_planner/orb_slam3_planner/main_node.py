#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import OccupancyGrid
import math
import numpy as np

from orb_slam3_planner.map_builder_module import MapBuilder
from orb_slam3_planner.planner_module import FrontierPlanner
from orb_slam3_planner.drone_controller_module import DroneController



class AutonomousExplorerNode(Node):
    """
    The central node that coordinates mapping, planning, and motion control for autonomous exploration.

    Responsibilities:
    - Subscribe to SLAM pose and point cloud
    - Maintain occupancy grid
    - Publish velocity, map, and visualization messages
    - Manage exploration state machine
    """

    def __init__(self):
        super().__init__('autonomous_explorer_node')


        # ======================
        # Namespace & Parameters
        # ======================
        self.declare_parameter('robot_namespace', '')
        self.robot_namespace = self.get_parameter('robot_namespace').value.rstrip('/')

        # ======================
        # Map Parameters
        # ======================
        self.cell_size = 0.25
        self.map_range = 20.0
        self.grid_size = int(2 * self.map_range / self.cell_size)

        self.occupancy_prob = np.full((self.grid_size, self.grid_size), 0.5, dtype=np.float32)
        self.update_count = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        self.height_min = 0.1
        self.height_max = 2.0
        self.obstacle_prob_increment = 0.2
        self.free_prob_decrement = -0.05
        self.occupied_threshold = 0.75
        self.free_threshold = 0.35
        self.freeze_update_count = 8

        # ======================
        # Sensor Parameters
        # ======================
        self.camera_fov = math.radians(60)
        self.camera_range = 10.0
        self.min_points_for_obstacle = 20

        # ======================
        # Motion Parameters
        # ======================
        self.linear_speed = 0.6
        self.angular_speed = 0.5
        self.safe_distance = 4

        self.adaptive_speed = True
        self.min_linear_speed = 0.3
        self.max_linear_speed = 0.8

        # ======================
        # Frontier Planning Parameters
        # ======================
        self.use_frontier_scoring = True
        self.visited_targets = set()
        self.exploration_radius = 2

        # ======================
        # Robot State
        # ======================
        self.robot_pos = None
        self.robot_angle = 0.0
        self.current_pose = None
        self.target = None
        self.state = "EXPLORING"

        self.collision_counter = 0
        self.stuck_counter = 0
        self.last_robot_pos = None

        # ======================
        # ROS2 Setup
        # ======================
        ns = f'/{self.robot_namespace}' if self.robot_namespace else ''

        self.cmd_pub = self.create_publisher(Twist, f'{ns}/cmd_vel', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, f'{ns}/planner_occupancy_grid', 10)
        self.goal_pub = self.create_publisher(Point, f'{ns}/goal_grid_pos', 10)
        self.robot_pos_pub = self.create_publisher(Point, f'{ns}/robot_grid_pos', 10)
        self.direction_pub = self.create_publisher(PoseStamped, f'{ns}/robot_direction', 10)

        self.mapper = MapBuilder(self)
        self.planner = FrontierPlanner(self)
        self.controller = DroneController(self)

        self.create_subscription(PointCloud2, f'{ns}/orb_slam3/landmarks_raw', self.mapper.pointcloud_callback, 10)
        self.create_subscription(PoseStamped, f'{ns}/robot_pose_slam', self.pose_callback, 10)

        self.create_timer(0.5, self.control_loop)

        self.get_logger().info(f"Autonomous Explorer Node started for namespace: {self.robot_namespace}")


    def update_cell_probability(self, x, y, prob_change):
        """
        Update a cell's occupancy probability using Bayesian-like updates.

        Args:
            x (int): Grid x index
            y (int): Grid y index
            prob_change (float): Positive for obstacles, negative for free space
        """
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return

        old_prob = self.occupancy_prob[y, x]

        if prob_change > 0:
            new_prob = old_prob + prob_change * (1 - old_prob)
        else:
            new_prob = old_prob + prob_change * old_prob

        self.occupancy_prob[y, x] = np.clip(new_prob, 0.01, 0.99)
        self.update_count[y, x] += 1

    def pose_callback(self, msg):
        """
        Update the robot's position and heading from incoming SLAM pose messages.

        Args:
            msg (geometry_msgs.msg.PoseStamped): The robot's current estimated pose.
        """
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

    def control_loop(self):
        """
        Main control loop: executes exploration state machine and sends movement commands.
        """

        if not self.robot_pos:
            self.controller.stop_robot()
            return

        if self.controller.is_stuck():
            self.get_logger().warn("Robot stuck. Switching to RECOVERY.")
            self.state = "RECOVERY"
            self.stuck_counter = 0

        self.last_robot_pos = self.robot_pos

        if self.state == "COLLISION_AVOIDANCE":
            if self.collision_counter > 0:
                twist = Twist()
                twist.angular.z = -self.angular_speed
                self.cmd_pub.publish(twist)
                self.collision_counter -= 1
            else:
                self.state = "EXPLORING"

        elif self.state == "RECOVERY":
            twist = Twist()
            twist.linear.x = -self.linear_speed * 0.8
            twist.angular.z = -self.angular_speed * 1.5
            self.cmd_pub.publish(twist)
            self.state = "EXPLORING"

        elif self.state == "EXPLORING":
            self.target = self.planner.find_nearest_frontier()

            if self.target:
                self.state = "MOVING_TO_TARGET"
                self.get_logger().info(f"New target: {self.target}")
                goal_msg = Point()
                goal_msg.x = float(self.target[0])
                goal_msg.y = float(self.target[1])
                self.goal_pub.publish(goal_msg)
            else:
                self.controller.turn_to_explore()

        elif self.state == "MOVING_TO_TARGET":
            if not self.target:
                self.state = "EXPLORING"
                return

            rx, ry = self.robot_pos
            tx, ty = self.target
            distance = math.sqrt((tx - rx) ** 2 + (ty - ry) ** 2)

            if distance < 2.0:
                self.get_logger().info("Reached target.")
                self.visited_targets.add((tx, ty))
                self.state = "EXPLORING"
                self.target = None
                self.controller.stop_robot()
                return

            if self.controller.check_collision_ahead():
                self.get_logger().warn("Obstacle ahead! Halting.")
                self.controller.stop_robot()
                return

            self.controller.move_toward_target()

    def normalize_angle(self, angle):
        """
        Normalize an angle to the range [-pi, pi].

        Args:
            angle (float): Angle in radians.

        Returns:
            float: Normalized angle.
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def publish_map(self):
        """
        Publish the occupancy grid map based on the current probability grid.
        """
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.resolution = self.cell_size
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size
        msg.info.origin.position.x = -self.map_range
        msg.info.origin.position.y = -self.map_range

        ros_grid = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                value = self.get_occupancy_value(x, y)
                if value == -1:
                    ros_grid.append(-1)
                elif value == 0:
                    ros_grid.append(0)
                else:
                    ros_grid.append(100)

        msg.data = ros_grid
        self.map_pub.publish(msg)

    def get_occupancy_value(self, x, y):
        """
        Convert a cell's probability to a discrete occupancy value.

        Args:
            x (int): Grid x coordinate.
            y (int): Grid y coordinate.

        Returns:
            int: 1 = occupied, 0 = free, -1 = unknown
        """
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return -1

        prob = self.occupancy_prob[y, x]
        updates = self.update_count[y, x]

        if updates < 2:
            return -1
        if prob > self.occupied_threshold:
            return 1
        if prob < self.free_threshold:
            return 0
        return -1


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
