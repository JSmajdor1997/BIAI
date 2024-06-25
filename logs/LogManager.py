import json
import os

class LogManager:
    def __init__(self, root_path):
        self.root_path = root_path
        self.log_file = os.path.join(root_path, 'logs.json')
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as file:
                json.dump({}, file)

    def update_value(self, key, value):
        with open(self.log_file, 'r') as file:
            data = json.load(file)
        data[key] = value
        with open(self.log_file, 'w') as file:
            json.dump(data, file, indent=4)

    def read_file_as_array(self):
        with open(self.log_file, 'r') as file:
            data = json.load(file)
        return list(data.values())