import glob, os, torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
this_dir = os.path.dirname(os.path.abspath(__file__))
requirements = ["torch>=1.4"]

_exts = os.listdir(this_dir)

_exts = [d for d in _exts if os.path.isdir(os.path.join(this_dir, d)) and d not in  ['build', 'ext.egg-info', '__pycache__']]

def make_cuda_ext(module):
    name = module + '_ext'
    src_path = os.path.join(this_dir, module, 'src')
    include_path = os.path.join(this_dir, module, 'include/')
    assert os.path.exists(src_path)

    ext_sources = glob.glob(os.path.join(src_path, "*.cpp"))
    ext_sources += glob.glob(os.path.join(src_path, "*.cc"))
    ext_sources += glob.glob(os.path.join(src_path, "*.cu"))

    if os.path.exists(include_path):
        ext_include_dirs = [include_path]  
    else:
        ext_include_dirs = []
   
    extra_compile_args = {
        'cxx': ['-w', '-std=c++14'],
        'nvcc': ['-w', '-std=c++14',
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__']}
    # module = 'models.modules.ext.{}'.format(module)
    return CUDAExtension(
        name='{}.{}'.format(module, name),
        define_macros= [('WITH_CUDA', None)],
        sources=ext_sources,
        include_dirs=ext_include_dirs,
        extra_compile_args=extra_compile_args)


    
if __name__ == '__main__':
    ext_modules = [
            make_cuda_ext(module=module) for module in _exts
        ]
    setup(
        name='ext3d',
        version='1.0.0',
        author='junl',
        author_email='junl@bit.edu.cn',
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension},
        requires= ['numpy','torch'], 
        packages=find_packages(), 
    )
                