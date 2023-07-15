import json

class JsonHander:
    @staticmethod
    def load(file_path, mode='rb'):
        with open(file_path, mode) as f:
            contents = json.load(f)
        return contents
    
    