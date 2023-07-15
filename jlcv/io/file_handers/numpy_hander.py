import numpy as np

class NumpyHander:
    @staticmethod
    def load(file_path):
        contents = np.load(file_path)
        return contents
    
