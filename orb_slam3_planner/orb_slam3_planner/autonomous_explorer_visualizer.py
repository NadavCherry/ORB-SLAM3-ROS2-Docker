#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
import numpy as np
import cv2
import os


class AutonomousExplorerVisualizer(Node):
    def __init__(self):
        super().__init__('autonomous_explorer_visualizer')

        # Folder for saving visualizations
        self.viz_path = '/tmp/autonomous_explorer_viz'
        os.makedirs(self.viz_path, exist_ok=True)

        # Subscribers
        self.create_subscription(OccupancyGrid, '/planner_occupancy_grid', self.map_callback, 10)
        self.create_subscription(Point, '/robot_grid_pos', self.robot_pos_callback, 10)
        self.create_subscription(Point, '/goal_grid_pos', self.goal_pos_callback, 10)

        self.robot_pos = None  # (x, y)
        self.goal_pos = None   # (x, y)

        self.frame_count = 0
        self.window_name = 'Autonomous Explorer Map'
        self.display_live = True
        self.scale = 7  # Bigger scale for better visibility

        self.get_logger().info('Autonomous Explorer Visualizer Node started')

    def robot_pos_callback(self, msg):
        self.robot_pos = (int(msg.x), int(msg.y))

    def goal_pos_callback(self, msg):
        self.goal_pos = (int(msg.x), int(msg.y))

    def map_callback(self, msg):
        width = msg.info.width
        height = msg.info.height

        grid_data = np.array(msg.data, dtype=np.int8).reshape((height, width))

        # Create base image (small)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[grid_data == -1] = (128, 128, 128)  # Unknown = Gray
        img[grid_data == 0] = (0, 255, 0)  # Free = Green
        img[grid_data == 100] = (0, 0, 255)  # Obstacle = Red

        img = cv2.flip(img, 0)

        # Scale the image first
        img_large = cv2.resize(img, (width * self.scale, height * self.scale), interpolation=cv2.INTER_NEAREST)

        # Scale positions
        if self.robot_pos:
            x, y = self.robot_pos
            x_large = x * self.scale * 20
            y_large = (height - 1 - y) * self.scale * 20
            cv2.circle(img_large, (x_large, y_large), radius=15, color=(255, 0, 0), thickness=-1)
            cv2.putText(img_large, 'Robot', (x_large + 10, y_large - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        if self.goal_pos:
            gx, gy = self.goal_pos
            gx_large = gx * self.scale * 10
            gy_large = (height - 1 - gy) * self.scale *10
            cv2.drawMarker(img_large, (gx_large, gy_large), color=(0, 255, 255), markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=30, thickness=4)
            cv2.putText(img_large, 'Goal', (gx_large + 10, gy_large - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255),
                        2)

        img_large = cv2.resize(img, (width * self.scale, height * self.scale), interpolation=cv2.INTER_NEAREST)

        filename = os.path.join(self.viz_path, f'grid_{self.frame_count:05d}.png')
        cv2.imwrite(filename, img_large)
        self.get_logger().info(f"Saved map visualization: {filename}")

        if self.display_live:
            cv2.imshow(self.window_name, img_large)
            cv2.waitKey(1)

        self.frame_count += 1

    def close(self):
        if self.display_live:
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousExplorerVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
