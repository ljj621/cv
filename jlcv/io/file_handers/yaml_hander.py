import yaml

class YamlHander:
    @staticmethod
    def load(file_path, mode='r'):
        with open(file_path, mode) as f:
            contents = yaml.safe_load(f)
        return contents
    @staticmethod
    def dump(contents, file_path):
        with open(file_path, 'w') as f:
            yaml.dump(contents, f, encoding='uft-8')