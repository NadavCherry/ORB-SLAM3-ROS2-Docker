#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2
import os


class AutonomousExplorerVisualizer(Node):
    def __init__(self):
        super().__init__('autonomous_explorer_visualizer')

        # Folder for saving visualizations
        self.viz_path = '/tmp/autonomous_explorer_viz'
        os.makedirs(self.viz_path, exist_ok=True)

        # Subscriber to occupancy grid published by explorer node
        self.create_subscription(
            OccupancyGrid,
            '/planner_occupancy_grid',
            self.map_callback,
            10
        )

        self.frame_count = 0
        self.window_name = 'Autonomous Explorer Map'
        self.display_live = True  # Set False to disable live display
        self.scale = 20  # BIGGER SCALE FACTOR (try 20 or more)

        self.get_logger().info('Autonomous Explorer Visualizer Node started')

    def map_callback(self, msg):
        width = msg.info.width
        height = msg.info.height

        grid_data = np.array(msg.data, dtype=np.int8).reshape((height, width))

        # Create RGB visualization: unknown=gray, free=green, occupied=red
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[grid_data == -1] = (128, 128, 128)  # Unknown = Gray
        img[grid_data == 0] = (0, 255, 0)       # Free = Green
        img[grid_data == 100] = (0, 0, 255)     # Obstacle = Red

        # Flip vertically to match map coordinates
        img = cv2.flip(img, 0)

        # Enlarge image significantly for better visibility
        img_large = cv2.resize(img, (width * self.scale, height * self.scale), interpolation=cv2.INTER_NEAREST)

        # Save to file
        filename = os.path.join(self.viz_path, f'grid_{self.frame_count:05d}.png')
        cv2.imwrite(filename, img_large)
        self.get_logger().info(f"Saved map visualization: {filename}")

        # Live display in big window
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
