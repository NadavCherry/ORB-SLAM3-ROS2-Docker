import struct
import rclpy
from rclpy.node import Node
from slam_msgs.srv import GetAllLandmarksInMap
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import time
import numpy as np
import std_msgs.msg

class MapMergerNode(Node):
    def __init__(self):
        super().__init__('map_merger_node')

        # List of robots with their initial positions
        self.robots = {
            0: [1.0, -0.5, 1.0], 
            1: [3.0, -2.0, 1.0],
            2: [5.0, -0.5, 0.0]
        }

        # Create a service client for each robot to get all landmarks in the map
        self.agents_clients = {}
        for robot_id in self.robots:
            service_name = f'/robot_{robot_id}/orb_slam3/get_all_landmarks_in_map'
            self.agents_clients[robot_id] = self.create_client(GetAllLandmarksInMap, service_name)

        # Ensure the services are available
        for robot_id, client in self.agents_clients.items():
            while not client.wait_for_service(timeout_sec=30.0):
                self.get_logger().info(f'Service for robot_{robot_id} not available, waiting...')
        
        # Create a publisher for the merged map (PointCloud2)
        self.map_publisher = self.create_publisher(PointCloud2, '/merged_map', 10)

        # To store the merged map
        self.merged_map = []

        self.last_landmarks = {robot_id: [] for robot_id in self.robots}

        self.get_logger().info('Map Merger Node started.')

        # Create a timer that requests the map data every 1 seconds
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        # For each robot, request the map data every 2 seconds
        for robot_id in self.robots:
            self.get_all_landmarks(robot_id)

    def get_all_landmarks(self, robot_id):
        # Create a service request for getting all landmarks in the map
        request = GetAllLandmarksInMap.Request()
        request.request = True  # Request the full map

        # Send the request asynchronously
        future = self.agents_clients[robot_id].call_async(request)
        future.add_done_callback(lambda future, robot_id=robot_id: self.handle_landmark_response(future, robot_id))

    def handle_landmark_response(self, future, robot_id):
        try:
            # Get the result from the service call
            response = future.result()
            
            if response:
                self.get_logger().info(f'Received {len(response.landmarks.data)} landmarks from robot_{robot_id}.')

                # Convert PointCloud2 data into a list of points
                points = list(pc2.read_points(response.landmarks, field_names=("x", "y", "z"), skip_nans=True))


                # Filter out invalid points (those with NaN values)
                valid_points = self.filter_invalid_points(points)
                
                if len(valid_points) != len(points):
                    self.get_logger().warn(f"Skipped {len(points) - len(valid_points)} invalid map points from robot_{robot_id}.")

                # Check for significant change in the map (you could use the same method for change detection here)
                if self.is_significant_change(robot_id, valid_points):
                    # Transform landmarks from robot's frame to a global frame
                    transformed_points = self.transform_landmarks_to_global(valid_points, robot_id)
                    
                    # Merge the points into the global map
                    self.merged_map.extend(transformed_points)
                    
                    # Publish the merged map
                    self.publish_merged_map()

                # Update the last landmarks for this robot
                self.last_landmarks[robot_id] = valid_points

            else:
                self.get_logger().info(f'No landmarks received from robot_{robot_id}.')
        except Exception as e:
            self.get_logger().error(f'Service call failed for robot_{robot_id}: {str(e)}')

    def filter_invalid_points(self, point_cloud):
        # Filter out invalid points (those with NaN values)
        valid_points = []
        for point in point_cloud:
            if not (np.isnan(point[0]) or np.isnan(point[1]) or np.isnan(point[2])):
                valid_points.append(point)
        return valid_points

    def publish_merged_map(self): 
        # Convert the merged map to a PointCloud2 message and publish it
        if not self.merged_map: # Check if map is empty
            self.get_logger().info('Merged map is empty, not publishing.')
            return

        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "base_footprint" # Ensure this frame_id is correct relative to your RViz fixed_frame
        
        pc2_msg = PointCloud2()
        pc2_msg.header = header
        pc2_msg.height = 1
        pc2_msg.width = len(self.merged_map)
        
        # Define point fields (x, y, z, rgb)
        pc2_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1)
        ]
        pc2_msg.is_bigendian = False
        pc2_msg.point_step = 16  # 3 floats (x, y, z) + 1 UINT32 (rgb)
        pc2_msg.row_step = pc2_msg.point_step * pc2_msg.width
        
        # Define RGB values for each robot (as 8-bit RGB values)
        # Using a list of tuples for ordered access, or keep dict if IDs are sparse
        colors = {
            0: [0, 0, 255],  # Blue (robot 0)
            1: [0, 255, 0],  # Green (robot 1)
            2: [255, 0, 0],  # Red (robot 2)
            # Add more colors for more robots if needed
            # 3: [255, 255, 0], # Yellow
            # 4: [0, 255, 255], # Cyan
        }
        
        points_data = bytearray() 

        # Add the points and color them based on their origin robot_id
        for point_data in self.merged_map: # Renamed point to point_data for clarity
            # point_data is now expected to be [x, y, z, original_robot_id]
            x, y, z = point_data[0], point_data[1], point_data[2]
            original_robot_id = point_data[3] # Extract the robot_id for this specific point

            # Get the RGB values for the specific robot that generated this point
            rgb = colors.get(original_robot_id, [128, 128, 128]) # Default to grey if ID not found
            
            # Pack the RGB values into a single 32-bit integer: 0xAARRGGBB
            rgb_value = (0xFF << 24) | (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
            
            # Pack the x, y, z as float32 and rgb_value as uint32
            packed_point = struct.pack('<fffI', x, y, z, rgb_value)
            points_data.extend(packed_point)

        pc2_msg.data = bytes(points_data) # Convert bytearray to bytes for the message
        
        # Publish the message
        self.map_publisher.publish(pc2_msg)
        self.get_logger().info(f'Merged map published with {len(self.merged_map)} points.')

    def is_significant_change(self, robot_id, new_points):
        # Check if the new points differ significantly from the last few points
        if len(self.last_landmarks[robot_id]) < 2:
            return True  # No previous points, always merge the new points

        # Calculate the Euclidean distance between the last points and the current ones
        distance_threshold = 0.2  # Example threshold for significant change

        for old_point, new_point in zip(self.last_landmarks[robot_id], new_points):
            distance = np.linalg.norm(np.array([old_point[0], old_point[1], old_point[2]]) -
                                    np.array([new_point[0], new_point[1], new_point[2]]))
            if distance > distance_threshold:
                return True

        return False

    def transform_landmarks_to_global(self, map_points, robot_id):
        # Transform the points to global coordinates based on robot's position (simple translation here)
        robot_position = np.array(self.robots[robot_id])
        transformed_points_with_id = [] # Now also storing robot_id

        for point in map_points:
            transformed_point_xyz = np.array([point[0], point[1], point[2]]) + robot_position
            # Append the original robot_id to the transformed point
            transformed_points_with_id.append([transformed_point_xyz[0], transformed_point_xyz[1], transformed_point_xyz[2], robot_id])
        
        return transformed_points_with_id

def main(args=None):
    rclpy.init(args=args)
    node = MapMergerNode()
    
    try:
        # Spin the node so the callback is called
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Shutdown the node
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
