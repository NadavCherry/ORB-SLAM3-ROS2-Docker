#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from slam_msgs.srv import GetAllLandmarksInMap
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String, Header
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np


class FilteredLandmarkPublisher(Node):
    def __init__(self):
        super().__init__('filtered_landmark_publisher')

        # Parameters for filtering
        self.declare_parameter('max_distance', 50.0)  # Maximum distance from origin
        self.declare_parameter('min_z', -2.0)  # Minimum Z (floor)
        self.declare_parameter('max_z', 5.0)  # Maximum Z (ceiling)

        self.max_distance = self.get_parameter('max_distance').value
        self.min_z = self.get_parameter('min_z').value
        self.max_z = self.get_parameter('max_z').value

        # Publishers
        self.raw_landmark_pub = self.create_publisher(PointCloud2, '/orb_slam3/landmarks_raw', 10)
        self.filtered_landmark_pub = self.create_publisher(PointCloud2, '/orb_slam3/landmarks_filtered', 10)
        self.status_pub = self.create_publisher(String, '/orb_slam3/landmark_status', 10)

        # Service client
        self.get_landmarks_client = self.create_client(GetAllLandmarksInMap, '/orb_slam3/get_all_landmarks_in_map')

        # Wait for service
        self.get_logger().info('Waiting for service...')
        self.get_landmarks_client.wait_for_service()
        self.get_logger().info('Service ready!')
        self.get_logger().info(
            f'Filtering parameters: max_distance={self.max_distance}, z_range=[{self.min_z}, {self.max_z}]')

        # Timer
        self.timer = self.create_timer(1.0, self.get_and_process_landmarks)

        self.last_raw_count = 0
        self.last_filtered_count = 0

    def get_and_process_landmarks(self):
        """Get landmarks and apply filtering"""
        request = GetAllLandmarksInMap.Request()
        request.request = True

        future = self.get_landmarks_client.call_async(request)
        future.add_done_callback(self.handle_landmark_response)

    def handle_landmark_response(self, future):
        """Process and filter landmarks"""
        try:
            response = future.result()

            if hasattr(response, 'landmarks'):
                raw_cloud = response.landmarks

                # Publish raw cloud
                self.raw_landmark_pub.publish(raw_cloud)

                # Extract points
                raw_points = list(pc2.read_points(raw_cloud, skip_nans=True))
                raw_count = len(raw_points)

                # Filter points
                filtered_points = []
                outlier_count = 0

                for point in raw_points:
                    x, y, z = point[0], point[1], point[2]

                    # Calculate distance from origin
                    distance = np.sqrt(x * x + y * y + z * z)

                    # Apply filters
                    if (distance <= self.max_distance and
                            z >= self.min_z and
                            z <= self.max_z):
                        filtered_points.append([x, y, z])
                    else:
                        outlier_count += 1

                filtered_count = len(filtered_points)

                # Create and publish filtered cloud
                if filtered_points:
                    header = Header()
                    header.stamp = self.get_clock().now().to_msg()
                    header.frame_id = raw_cloud.header.frame_id

                    filtered_cloud = pc2.create_cloud_xyz32(header, filtered_points)
                    self.filtered_landmark_pub.publish(filtered_cloud)

                # Update status
                status_msg = String()
                status_msg.data = f"Raw: {raw_count}, Filtered: {filtered_count}, Outliers: {outlier_count}"
                self.status_pub.publish(status_msg)

                # Log changes
                if raw_count != self.last_raw_count or filtered_count != self.last_filtered_count:
                    self.get_logger().info(
                        f'Points - Raw: {raw_count}, Filtered: {filtered_count}, Removed: {outlier_count}')

                    if filtered_points:
                        # Calculate statistics for filtered points
                        points_array = np.array(filtered_points)

                        self.get_logger().info('Filtered point cloud statistics:')
                        self.get_logger().info(
                            f'  X range: [{np.min(points_array[:, 0]):.3f}, {np.max(points_array[:, 0]):.3f}]')
                        self.get_logger().info(
                            f'  Y range: [{np.min(points_array[:, 1]):.3f}, {np.max(points_array[:, 1]):.3f}]')
                        self.get_logger().info(
                            f'  Z range: [{np.min(points_array[:, 2]):.3f}, {np.max(points_array[:, 2]):.3f}]')

                        # # Save filtered points
                        # if filtered_count != self.last_filtered_count:
                        #     self.save_filtered_landmarks(filtered_points)

                    self.last_raw_count = raw_count
                    self.last_filtered_count = filtered_count

        except Exception as e:
            self.get_logger().error(f'Error: {e}')

    def save_filtered_landmarks(self, points):
        """Save filtered landmarks to file"""
        filename = f'orb_slam3_landmarks_filtered_{self.get_clock().now().nanoseconds}.txt'
        with open(filename, 'w') as f:
            f.write('# Filtered ORB-SLAM3 Landmarks\n')
            f.write(f'# Filter: max_distance={self.max_distance}, z_range=[{self.min_z}, {self.max_z}]\n')
            f.write('# x, y, z\n')
            f.write(f'# Total points: {len(points)}\n')
            for p in points:
                f.write(f'{p[0]}, {p[1]}, {p[2]}\n')
        self.get_logger().info(f'Saved {len(points)} filtered landmarks to {filename}')


def main(args=None):
    rclpy.init(args=args)
    node = FilteredLandmarkPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
