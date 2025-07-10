import numpy as np
import math
import sensor_msgs_py.point_cloud2 as pc2

class MapBuilder:
    """
    MapBuilder is responsible for constructing and maintaining a probabilistic 2D occupancy grid map
    based on incoming 3D point cloud data from sensors.
    """

    def __init__(self, node):
        """
        Initialize the MapBuilder with references to the main ROS2 node and its shared state.

        Args:
            node (rclpy.node.Node): The main ROS2 node containing shared parameters and utilities.
        """
        self.node = node

        # Shortcuts to main node utilities (only functions or fixed values)
        self.get_logger = node.get_logger
        self.get_clock = node.get_clock
        self.publish_map = node.publish_map
        self.update_cell_probability = node.update_cell_probability
        self.get_logger().info("MapBuilder initialized")

    def pointcloud_callback(self, msg):
        """
        Callback function for processing incoming 3D point cloud data.
        Updates the occupancy grid probabilities based on observed obstacles and free space.

        Args:
            msg (sensor_msgs.msg.PointCloud2): Incoming raw point cloud message.
        """
        if self.node.current_pose is None:
            return

        robot_world_x = self.node.current_pose.position.x
        robot_world_y = self.node.current_pose.position.y

        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        cell_points = {}

        for x, y, z in points:
            if not (self.node.height_min <= z <= self.node.height_max):
                continue

            angle_to_point = math.atan2(y - robot_world_y, x - robot_world_x)
            angle_diff = self.node.normalize_angle(angle_to_point - self.node.robot_angle)

            if abs(angle_diff) > self.node.camera_fov / 2:
                continue

            dist = math.sqrt((x - robot_world_x) ** 2 + (y - robot_world_y) ** 2)
            if dist > self.node.camera_range:
                continue

            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue

            grid_x = int((x + self.node.map_range) / self.node.cell_size)
            grid_y = int((y + self.node.map_range) / self.node.cell_size)

            if 0 <= grid_x < self.node.grid_size and 0 <= grid_y < self.node.grid_size:
                cell_points.setdefault((grid_x, grid_y), []).append((x, y, z))

        occupied_cells = set()

        for (gx, gy), pts in cell_points.items():
            if len(pts) >= self.node.min_points_for_obstacle:
                prob_increase = min(self.node.obstacle_prob_increment * len(pts), 0.5)
                self.update_cell_probability(gx, gy, prob_increase)
                occupied_cells.add((gx, gy))

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.node.grid_size and 0 <= ny < self.node.grid_size:
                            self.update_cell_probability(nx, ny, prob_increase * 0.5)

        if self.node.robot_pos:
            self.update_free_space_probability(occupied_cells)

        self.decay_probabilities()
        self.publish_map()

    def decay_probabilities(self):
        """
        Apply slow decay to all occupancy grid cells that haven't been updated,
        returning their probabilities toward the neutral 0.5 value over time.
        """
        decay_factor = 0.99

        for y in range(self.node.grid_size):
            for x in range(self.node.grid_size):
                if self.node.update_count[y, x] < self.node.freeze_update_count:
                    old_prob = self.node.occupancy_prob[y, x]
                    self.node.occupancy_prob[y, x] = 0.5 + (old_prob - 0.5) * decay_factor

    def bresenham_line(self, start_x, start_y, end_x, end_y):
        """
        Bresenham's line algorithm: Generate grid cells along a line.

        Returns:
            list of (int, int): List of grid cells along the line.
        """
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
            if 0 <= x < self.node.grid_size and 0 <= y < self.node.grid_size:
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

    def update_free_space_probability(self, occupied_cells):
        """
        Perform ray tracing from the robot to observed obstacles to mark intervening space as free.

        Args:
            occupied_cells (set of (int, int)): Set of grid coordinates identified as occupied.
        """
        if not self.node.robot_pos:
            return

        robot_gx, robot_gy = self.node.robot_pos

        for gx, gy in occupied_cells:
            dx = gx - robot_gx
            dy = gy - robot_gy
            angle_to_obstacle = math.atan2(dy, dx)
            angle_diff = self.node.normalize_angle(angle_to_obstacle - self.node.robot_angle)

            if abs(angle_diff) <= self.node.camera_fov / 2:
                cells_on_ray = self.bresenham_line(robot_gx, robot_gy, gx, gy)
                for (x, y) in cells_on_ray[:-1]:
                    if 0 <= x < self.node.grid_size and 0 <= y < self.node.grid_size:
                        self.update_cell_probability(x, y, self.node.free_prob_decrement * 1.5)

        fov_half = self.node.camera_fov / 2
        num_rays = int(self.node.camera_fov / math.radians(5))
        max_range_cells = int(self.node.camera_range / self.node.cell_size)

        for i in range(num_rays):
            angle_offset = -fov_half + (i * self.node.camera_fov / (num_rays - 1))
            angle = self.node.robot_angle + angle_offset

            for dist in range(1, max_range_cells):
                x = int(robot_gx + dist * math.cos(angle))
                y = int(robot_gy + dist * math.sin(angle))

                if not (0 <= x < self.node.grid_size and 0 <= y < self.node.grid_size):
                    break

                if self.node.occupancy_prob[y, x] > self.node.occupied_threshold and self.node.update_count[y, x] > 3:
                    break

                self.update_cell_probability(x, y, self.node.free_prob_decrement)
