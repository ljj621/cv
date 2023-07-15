import pickle

class PickleHander:
    @staticmethod
    def load(file_path, mode='rb'):
        with open(file_path, mode) as f:
            contents = pickle.load(f)
        return contents
    
    @staticmethod
    def dump(contents, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(contents, f)