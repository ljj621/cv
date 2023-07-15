import numpy as np

class BinHander:
    @staticmethod
    def load(file_path, dtype, do_unpack=False):
        contents = np.fromfile(file_path, dtype=dtype)
        if not do_unpack: return contents

        uncompressed = np.zeros(contents.shape[0] * 8, dtype=np.uint8)
        uncompressed[::8] = contents[:] >> 7 & 1
        uncompressed[1::8] = contents[:] >> 6 & 1
        uncompressed[2::8] = contents[:] >> 5 & 1
        uncompressed[3::8] = contents[:] >> 4 & 1
        uncompressed[4::8] = contents[:] >> 3 & 1
        uncompressed[5::8] = contents[:] >> 2 & 1
        uncompressed[6::8] = contents[:] >> 1 & 1
        uncompressed[7::8] = contents[:] & 1
        return contents
    
