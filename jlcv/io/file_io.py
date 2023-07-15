import os
import copy
from .file_handers import FileHander

class FileIO(object):
    @classmethod
    def load(cls, file_path, *args, **kwargs):
        endwith = file_path.split('.')[-1]
        hander = FileHander[endwith]

        contents = hander.load(file_path, *args, **kwargs)
        return contents

    @classmethod
    def dump(cls, contents, file_path):
        endwith = file_path.split('.')[-1]
        hander = FileHander[endwith]
        
        hander.dump(contents, file_path)

    

            




    

    


