
class Task:
    def __init__(self, task_id, task_node):
        self.task_id = task_id
        self.task_node = task_node
        self.over_time = None

    def is_over(self):
        return self.over_time is not None

    def complete_task(self, clock):
        self.over_time = clock


class Vehicle:
    def __init__(self, start_node):
        self.cur_node = start_node
        self.start_node = start_node
        self.end_node = None

        self.trajectory = []
        self.previous_paths = []  # Store historical paths (up to 3 paths)

        self.distance_cost = 0.0
        self.time_cost = 0.0
        self.speed = 45000.0/3600.0  # m/s

    def set_end_node(self, node):
        self.end_node = node

