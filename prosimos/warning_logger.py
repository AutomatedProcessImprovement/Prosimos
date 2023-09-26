class WarningLogger:
    def __init__(self):
        self.warnings_queue = []

    def add_warning(self, warning_message):
        self.warnings_queue.append(warning_message)

    def add_warnings(self, warning_messages):
        self.warnings_queue.extend(warning_messages)

    def pop_warning(self):
        if not self.is_empty():
            return self.warnings_queue.pop(0)
        return None

    def is_empty(self):
        return len(self.warnings_queue) == 0

    def get_all_warnings(self):
        return self.warnings_queue.copy()

    def clear_warnings(self):
        self.warnings_queue.clear()

    def display_all_warnings(self):
        for warning in self.warnings_queue:
            print(warning)


warning_logger = WarningLogger()
