import h5py
import numpy as np

class H5pyHander:
    @staticmethod
    def load(file_path, mode='r'):
        contents = {}
        with h5py.File(file_path, mode) as f:
            for k in f.keys():
                contents[k] = np.array(f[k])
        return contents
    
    @staticmethod
    def dump(contents, file_path):
        with h5py.File(file_path, 'w') as f:
            if isinstance(contents, dict):
                for k, v in contents.items():
                    f[k] = v

            
