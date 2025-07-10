import math

class FrontierPlanner:
    """
    FrontierPlanner is responsible for identifying unexplored frontiers in the occupancy grid
    and selecting optimal exploration targets based on distance, novelty, information gain,
    and alignment with the robot's heading.
    """

    def __init__(self, node):
        """
        Initialize the FrontierPlanner with access to shared state from the main node.

        Args:
            node (rclpy.node.Node): The central node providing shared environment and robot data.
        """
        self.node = node
        self.get_logger = node.get_logger
        self.get_occupancy_value = node.get_occupancy_value
        self.normalize_angle = node.normalize_angle

        self.grid_size = node.grid_size
        self.occupied_threshold = node.occupied_threshold

        self.get_logger().info("FrontierPlanner initialized")

    def find_frontiers(self):
        """
        Scan the occupancy grid to identify frontiers—cells that are known to be free but border unknown space.

        Returns:
            list of (int, int): Grid coordinates of frontier cells.
        """
        frontiers = []

        for y in range(1, self.grid_size - 1):
            for x in range(1, self.grid_size - 1):
                if self.get_occupancy_value(x, y) != 0:
                    continue  # Only consider free cells

                has_unknown_neighbor = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if self.get_occupancy_value(x + dx, y + dy) == -1:
                        has_unknown_neighbor = True
                        break

                if has_unknown_neighbor:
                    frontiers.append((x, y))

        return frontiers

    def find_best_frontier(self):
        """
        Select the optimal frontier using a multi-factor scoring system.
        Returns None if no valid frontiers are found.

        Returns:
            (int, int) or None: Grid coordinates of the best frontier.
        """

        if not self.node.robot_pos:
            return None

        frontiers = self.find_frontiers()

        if not frontiers:
            self.get_logger().warn("No frontiers found — stopping exploration")
            return None

        rx, ry = self.node.robot_pos
        best_frontier = None
        best_score = float('inf')

        for fx, fy in frontiers:
            if not self.is_safe_position(fx, fy):
                continue

            distance = math.sqrt((fx - rx) ** 2 + (fy - ry) ** 2)
            if distance < 3:
                continue
            if distance > 30:
                continue

            score = self.calculate_frontier_score(fx, fy, rx, ry)

            if score < best_score:
                best_score = score
                best_frontier = (fx, fy)

        return best_frontier

    def find_nearest_frontier(self):
        """
        Wrapper function to find a frontier based either on scoring or pure distance.

        Returns:
            (int, int) or None: Grid coordinates of the chosen frontier.
        """

        if self.node.use_frontier_scoring:
            return self.find_best_frontier()

        if not self.node.robot_pos:
            return None

        frontiers = self.find_frontiers()

        if not frontiers:
            self.get_logger().warn("No frontiers found — exploration halted")
            return None

        rx, ry = self.node.robot_pos
        best_frontier = None
        min_distance = float('inf')

        for fx, fy in frontiers:
            distance = math.sqrt((fx - rx) ** 2 + (fy - ry) ** 2)

            if not self.is_safe_position(fx, fy):
                continue

            if distance < 3:
                continue

            if distance > 20:
                continue


            if distance < min_distance:
                min_distance = distance
                best_frontier = (fx, fy)

        return best_frontier

    def calculate_frontier_score(self, fx, fy, rx, ry):
        """
        Compute a score for a given frontier based on distance, novelty, surrounding unknowns, and heading alignment.

        Returns:
            float: Calculated score for the frontier.
        """
        distance = math.sqrt((fx - rx) ** 2 + (fy - ry) ** 2)

        novelty_bonus = 1.0
        for vx, vy in self.node.visited_targets:
            if math.sqrt((fx - vx) ** 2 + (fy - vy) ** 2) < self.node.exploration_radius:
                novelty_bonus = 0.3
                break

        unknown_count = 0
        check_radius = 3
        for dx in range(-check_radius, check_radius + 1):
            for dy in range(-check_radius, check_radius + 1):
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.get_occupancy_value(nx, ny) == -1:
                        unknown_count += 1

        angle_to_frontier = math.atan2(fy - ry, fx - rx)
        angle_diff = abs(self.normalize_angle(angle_to_frontier - self.node.robot_angle))
        angle_factor = 1.0 - (angle_diff / math.pi) * 0.3

        score = distance / novelty_bonus / (1 + unknown_count * 0.1) / angle_factor
        return score

    def is_safe_position(self, x, y):
        """
        Determine if the specified position is safe by checking for nearby obstacles.

        Returns:
            bool: True if the area is free from obstacles; False otherwise.
        """
        for dx in range(-self.node.safe_distance, self.node.safe_distance + 1):
            for dy in range(-self.node.safe_distance, self.node.safe_distance + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.node.occupancy_prob[ny, nx] > self.occupied_threshold:
                        return False
        return True
