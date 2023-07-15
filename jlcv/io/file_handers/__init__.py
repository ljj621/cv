from .h5py_hander import H5pyHander
from .json_hander import JsonHander
from .pickle_hander import PickleHander
from .txt_hander import TxtHander
from .yaml_hander import YamlHander
from .bin_hander import BinHander
from .numpy_hander import NumpyHander

FileHander = {
    'txt': TxtHander,
    'pkl': PickleHander,
    'yaml': YamlHander,
    'h5': H5pyHander,
    'json': JsonHander,
    'bin': BinHander,
    'list': TxtHander,
    'npy': NumpyHander
}

__all__ = [
    'FileHander'
]