#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
import numpy as np
import cv2


class AutonomousExplorerVisualizer(Node):
    def __init__(self):
        super().__init__('autonomous_explorer_visualizer')

        # Visualization parameters
        self.scale = 7  # Scale factor for display (same as original)
        self.window_name = 'Autonomous Explorer Map'

        # Robot and goal positions
        self.robot_pos = None
        self.robot_dir = None
        self.goal_pos = None

        # Subscribers - use your original topic names
        self.create_subscription(OccupancyGrid, '/planner_occupancy_grid', self.map_callback, 10)
        self.create_subscription(Point, '/goal_grid_pos', self.goal_callback, 10)
        self.create_subscription(Point, '/robot_grid_pos', self.robot_pos_callback, 10)

        self.get_logger().info('Autonomous Explorer Visualizer Node started')

    def robot_pos_callback(self, msg):
        """Update robot position and direction"""
        self.robot_pos = (int(msg.x), int(msg.y))
        self.robot_dir = float(msg.z)

    def goal_callback(self, msg):
        """Update goal position"""
        self.goal_pos = (int(msg.x), int(msg.y))

    def map_callback(self, msg):
        """Visualize the map with robot and goal"""
        width = msg.info.width
        height = msg.info.height

        # Convert occupancy grid to numpy array
        grid_data = np.array(msg.data, dtype=np.int8).reshape((height, width))

        # Create colored image
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Color coding
        img[grid_data == -1] = (128, 128, 128)  # Unknown = Gray
        img[grid_data == 0] = (0, 255, 0)  # Free = Green
        img[grid_data == 100] = (0, 0, 255)  # Obstacle = Red

        # Flip image (ROS uses different coordinate system)
        img = cv2.flip(img, 0)

        # Scale up for better visibility
        img_large = cv2.resize(img, (width * self.scale, height * self.scale),
                               interpolation=cv2.INTER_NEAREST)

        # Draw robot position and direction
        if self.robot_pos:
            rx, ry = self.robot_pos
            if 0 <= rx < width and 0 <= ry < height:
                rx_large = rx * self.scale
                ry_large = (height - 1 - ry) * self.scale

                # Draw robot as blue circle
                cv2.circle(img_large, (rx_large, ry_large),
                           radius=max(5, self.scale * 1), color=(255, 0, 0), thickness=-1)

                # Draw direction arrow (continuous 360°)
                if self.robot_dir is not None:
                    arrow_len = self.scale * 3  # Longer arrow
                    dx = int(arrow_len * math.cos(self.robot_dir))
                    dy = int(-arrow_len * math.sin(self.robot_dir))  # minus because of image flip

                    cv2.arrowedLine(img_large, (rx_large, ry_large),
                                    (rx_large + dx, ry_large + dy),
                                    (0, 255, 255), 2, tipLength=0.4)

        # Draw goal position
        if self.goal_pos:
            gx, gy = self.goal_pos
            if 0 <= gx < width and 0 <= gy < height:
                gx_large = gx * self.scale
                gy_large = (height - 1 - gy) * self.scale

                # Draw goal as yellow circle
                cv2.circle(img_large, (gx_large, gy_large),
                           radius=self.scale * 2, color=(0, 255, 255), thickness=2)

                # Add text label
                cv2.putText(img_large, 'GOAL',
                            (gx_large + 10, gy_large - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Add title and info
        cv2.putText(img_large, 'Autonomous Explorer Map',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(img_large, 'Green=Free, Red=Obstacle, Gray=Unknown, Blue=Robot',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Show robot position
        if self.robot_pos:
            cv2.putText(img_large, f'Robot: ({self.robot_pos[0]}, {self.robot_pos[1]})',
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        if self.goal_pos:
            cv2.putText(img_large, f'Goal: ({self.goal_pos[0]}, {self.goal_pos[1]})',
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Display
        cv2.imshow(self.window_name, img_large)
        cv2.waitKey(1)

    def destroy_node(self):
        """Clean up on shutdown"""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousExplorerVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()