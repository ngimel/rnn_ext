
from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import sys
if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for gnmt.')
#with open('requirements.txt') as f:
#        reqs = f.read()
rnn_utils = CUDAExtension(
                        name='gemm_C',
                        sources=['rnn_ext.cpp', 'rnn_ext_kernel.cu'],
                        extra_compile_args={
                                'cxx': ['-O2',],
                                'nvcc':['--gpu-architecture=sm_70',]
                        }
)
setup(
    name='rnn',
    description='RNN',
#    install_requires=reqs.strip().split('\n'),
    packages=find_packages(),
    ext_modules=[rnn_utils],
    cmdclass={
                'build_ext': BuildExtension
    },
)
