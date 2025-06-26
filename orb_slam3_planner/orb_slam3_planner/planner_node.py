#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import threading
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull


class OrbSlam3Planner(Node):
    """
    Node that receives 3D landmarks from ORB-SLAM3 and creates a 2D occupancy grid.
    Uses clustering and convex hull to identify obstacles from sparse landmarks.
    """

    def __init__(self):
        super().__init__('orb_slam3_planner')

        # Parameters
        self.declare_parameter('resolution', 0.1)  # Grid resolution in meters
        self.declare_parameter('map_size', 40.0)  # Map size in meters
        self.declare_parameter('z_min', -0.5)  # Min height relative to robot
        self.declare_parameter('z_max', 2.0)  # Max height relative to robot
        self.declare_parameter('cluster_eps', 0.5)  # DBSCAN clustering distance
        self.declare_parameter('min_cluster_size', 3)  # Min points per cluster
        self.declare_parameter('obstacle_inflation', 0.3)  # Obstacle inflation radius

        self.resolution = self.get_parameter('resolution').value
        self.map_size = self.get_parameter('map_size').value
        self.z_min = self.get_parameter('z_min').value
        self.z_max = self.get_parameter('z_max').value
        self.cluster_eps = self.get_parameter('cluster_eps').value
        self.min_cluster_size = self.get_parameter('min_cluster_size').value
        self.obstacle_inflation = self.get_parameter('obstacle_inflation').value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid_2d', 10)

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, '/robot_pose_slam', self.pose_callback, 10)
        self.landmark_sub = self.create_subscription(
            PointCloud2, '/orb_slam3/landmarks_raw', self.landmark_callback, qos)

        # State
        self.current_pose = None
        self.latest_landmarks = None
        self.landmark_count = 0
        self.lock = threading.Lock()

        # Accumulated landmarks for better map building
        self.accumulated_landmarks = []
        self.max_accumulated_points = 5000  # Limit memory usage

        # Timer for periodic map updates
        self.timer = self.create_timer(0.5, self.update_map)

        self.get_logger().info('ORB-SLAM3 Planner initialized with parameters:')
        self.get_logger().info(f'  Resolution: {self.resolution} m')
        self.get_logger().info(f'  Map size: {self.map_size} m')
        self.get_logger().info(f'  Height filter: [{self.z_min}, {self.z_max}] m')
        self.get_logger().info(f'  Clustering: eps={self.cluster_eps}, min_size={self.min_cluster_size}')

    def pose_callback(self, msg):
        """Store the latest robot pose"""
        with self.lock:
            self.current_pose = msg

    def landmark_callback(self, msg):
        """Process incoming 3D landmark point cloud"""
        with self.lock:
            self.latest_landmarks = msg
            try:
                points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
                self.landmark_count = len(points)

                if self.landmark_count > 0:
                    # Add new landmarks to accumulated set
                    self.accumulated_landmarks.extend(points)

                    # Limit accumulated points to prevent memory issues
                    if len(self.accumulated_landmarks) > self.max_accumulated_points:
                        # Keep only the most recent points
                        self.accumulated_landmarks = self.accumulated_landmarks[-self.max_accumulated_points:]

                    self.get_logger().info(
                        f"Received {self.landmark_count} new landmarks, "
                        f"total accumulated: {len(self.accumulated_landmarks)}")

            except Exception as e:
                self.get_logger().error(f"Error reading landmarks: {e}")

    def update_map(self):
        """Periodically update the occupancy grid from accumulated landmarks"""
        with self.lock:
            if self.current_pose is None:
                self.get_logger().warn("No pose received yet")
                return

            if len(self.accumulated_landmarks) == 0:
                self.get_logger().warn("No landmarks accumulated yet")
                return

            self.process_landmarks_to_occupancy_grid()

    def process_landmarks_to_occupancy_grid(self):
        """Create 2D occupancy grid from 3D landmarks using clustering"""

        # Get robot height for relative filtering
        robot_z = self.current_pose.pose.position.z if self.current_pose else 0.0

        # Filter landmarks by height and distance
        filtered_points = []
        for point in self.accumulated_landmarks:
            x, y, z = point[0], point[1], point[2]

            # Height filter relative to robot
            dz = z - robot_z
            if self.z_min <= dz <= self.z_max:
                # Distance filter
                distance = np.sqrt(x ** 2 + y ** 2)
                if distance <= self.map_size / 2:
                    filtered_points.append([x, y])

        if len(filtered_points) < self.min_cluster_size:
            self.get_logger().warn(
                f"Only {len(filtered_points)} points passed filters, need at least {self.min_cluster_size}")
            return

        points_2d = np.array(filtered_points)

        # Create occupancy grid
        grid_size = int(self.map_size / self.resolution)
        grid = np.ones((grid_size, grid_size), dtype=np.float32) * -1  # Unknown = -1

        # Convert points to grid coordinates
        grid_points = self.world_to_grid(points_2d, grid_size)

        # Cluster points to identify obstacles
        if len(grid_points) >= self.min_cluster_size:
            clustering = DBSCAN(eps=self.cluster_eps / self.resolution,
                                min_samples=self.min_cluster_size).fit(grid_points)
            labels = clustering.labels_

            # Process each cluster
            for label in set(labels):
                if label == -1:  # Noise points
                    continue

                cluster_points = grid_points[labels == label]

                if len(cluster_points) >= 3:
                    try:
                        # Create convex hull for the cluster
                        hull = ConvexHull(cluster_points)
                        hull_points = cluster_points[hull.vertices]

                        # Fill the convex hull as occupied
                        self.fill_polygon(grid, hull_points, value=100)  # Occupied = 100

                        # Inflate obstacles for safety
                        if self.obstacle_inflation > 0:
                            kernel_size = int(self.obstacle_inflation / self.resolution)
                            if kernel_size > 0:
                                kernel = cv2.getStructuringElement(
                                    cv2.MORPH_ELLIPSE, (kernel_size * 2 + 1, kernel_size * 2 + 1))
                                grid_occupied = (grid == 100).astype(np.uint8)
                                inflated = cv2.dilate(grid_occupied, kernel)
                                grid[inflated > 0] = 100

                    except Exception as e:
                        # If convex hull fails, just mark individual points
                        for pt in cluster_points:
                            x, y = int(pt[0]), int(pt[1])
                            if 0 <= x < grid_size and 0 <= y < grid_size:
                                grid[y, x] = 100

        # Mark free space around the robot
        if self.current_pose:
            robot_grid = self.world_to_grid(
                np.array([[self.current_pose.pose.position.x,
                           self.current_pose.pose.position.y]]), grid_size)
            if len(robot_grid) > 0:
                rx, ry = int(robot_grid[0][0]), int(robot_grid[0][1])
                # Clear area around robot
                radius = int(1.0 / self.resolution)  # 1 meter radius
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dx ** 2 + dy ** 2 <= radius ** 2:
                            gx, gy = rx + dx, ry + dy
                            if 0 <= gx < grid_size and 0 <= gy < grid_size:
                                grid[gy, gx] = 0  # Free = 0

        # Publish occupancy grid
        self.publish_occupancy_grid(grid)

        # Save visualization
        self.save_grid_visualization(grid, grid_points)

    def world_to_grid(self, world_points, grid_size):
        """Convert world coordinates to grid coordinates"""
        # Center the grid at (0, 0)
        grid_points = (world_points + self.map_size / 2) / self.resolution

        # Clip to grid bounds
        grid_points = np.clip(grid_points, 0, grid_size - 1)

        return grid_points.astype(int)

    def fill_polygon(self, grid, vertices, value):
        """Fill a polygon in the grid with the specified value"""
        # Convert to integer coordinates
        vertices = vertices.astype(np.int32)

        # Create a mask for the polygon
        mask = np.zeros_like(grid, dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], 1)

        # Fill the grid
        grid[mask == 1] = value

    def publish_occupancy_grid(self, grid):
        """Publish the occupancy grid as ROS message"""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.info.resolution = self.resolution
        msg.info.width = grid.shape[1]
        msg.info.height = grid.shape[0]

        # Origin at bottom-left corner, centered at (0, 0)
        msg.info.origin.position.x = -self.map_size / 2
        msg.info.origin.position.y = -self.map_size / 2
        msg.info.origin.position.z = 0.0

        # Convert to int8 format expected by ROS
        # -1 = unknown, 0 = free, 100 = occupied
        grid_int8 = np.zeros_like(grid, dtype=np.int8)
        grid_int8[grid < 0] = -1
        grid_int8[grid == 0] = 0
        grid_int8[grid > 50] = 100

        # Flatten in row-major order
        msg.data = grid_int8.flatten().tolist()

        self.map_pub.publish(msg)
        self.get_logger().info(
            f"Published occupancy grid {msg.info.width}x{msg.info.height}")

    def save_grid_visualization(self, grid, landmark_points):
        """Save visualization of the occupancy grid"""
        # Create RGB visualization
        vis = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)

        # Unknown = gray
        vis[grid < 0] = [128, 128, 128]

        # Free = white
        vis[grid == 0] = [255, 255, 255]

        # Occupied = black
        vis[grid > 50] = [0, 0, 0]

        # Draw landmarks as red dots
        for pt in landmark_points:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)

        # Draw robot position if available
        if self.current_pose:
            robot_grid = self.world_to_grid(
                np.array([[self.current_pose.pose.position.x,
                           self.current_pose.pose.position.y]]), grid.shape[0])
            if len(robot_grid) > 0:
                rx, ry = int(robot_grid[0][0]), int(robot_grid[0][1])
                cv2.circle(vis, (rx, ry), 5, (0, 255, 0), -1)
                cv2.circle(vis, (rx, ry), 5, (0, 128, 0), 2)

        # Save image
        filename = '/tmp/occupancy_grid_visualization.png'
        success = cv2.imwrite(filename, vis)
        if success:
            self.get_logger().info(f"Successfully saved grid visualization to {filename}")
            # Also save with timestamp for debugging
            import time
            timestamp = int(time.time())
            timestamped_filename = f'/tmp/occupancy_grid_{timestamp}.png'
            cv2.imwrite(timestamped_filename, vis)
            self.get_logger().info(f"Also saved to {timestamped_filename}")
        else:
            self.get_logger().error(f"Failed to save grid visualization to {filename}")


def main(args=None):
    rclpy.init(args=args)
    node = OrbSlam3Planner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()