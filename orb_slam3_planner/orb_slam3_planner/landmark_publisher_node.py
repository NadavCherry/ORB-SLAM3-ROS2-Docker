#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from slam_msgs.srv import GetAllLandmarksInMap
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import sensor_msgs_py.point_cloud2 as pc2


class RawLandmarkPublisher(Node):
    """
    Node that requests all 3D landmarks from ORB-SLAM3 and publishes them without filtering.
    """

    def __init__(self):
        super().__init__('raw_landmark_publisher')

        # Declare namespace parameter
        self.declare_parameter('robot_namespace', '')
        self.robot_namespace = self.get_parameter('robot_namespace').value.rstrip('/')
        ns = f'/{self.robot_namespace}' if self.robot_namespace else ''


        # Publisher
        self.raw_landmark_pub = self.create_publisher(PointCloud2, f'{ns}/orb_slam3/landmarks_raw', 10)
        self.status_pub = self.create_publisher(String, f'{ns}/orb_slam3/landmark_status', 10)

        # Service client
        service_name = f'{ns}/orb_slam3/get_all_landmarks_in_map'
        self.get_landmarks_client = self.create_client(GetAllLandmarksInMap, service_name)

        self.get_logger().info(f'Waiting for ORB-SLAM3 landmark service: {service_name}')
        self.get_landmarks_client.wait_for_service()
        self.get_logger().info('Service ready!')

        # Timer to fetch landmarks periodically
        self.timer = self.create_timer(1.0, self.fetch_and_publish_landmarks)

        self.last_raw_count = 0

    def fetch_and_publish_landmarks(self):
        """Send async request to get all raw landmarks"""
        request = GetAllLandmarksInMap.Request()
        request.request = True
        future = self.get_landmarks_client.call_async(request)
        future.add_done_callback(self.handle_landmark_response)

    def handle_landmark_response(self, future):
        """Publish the raw landmark point cloud"""
        try:
            response = future.result()
            cloud = response.landmarks
            self.raw_landmark_pub.publish(cloud)

            # Count and report how many points received
            points = list(pc2.read_points(cloud, skip_nans=True))
            raw_count = len(points)

            if raw_count != self.last_raw_count:
                status = String()
                status.data = f"Raw landmarks: {raw_count}"
                self.status_pub.publish(status)
                # self.get_logger().info(status.data)
                self.last_raw_count = raw_count

        except Exception as e:
            self.get_logger().error(f"Error receiving landmarks: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RawLandmarkPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
