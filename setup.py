from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
        ]
        
        build_args = ['--config', 'Release']
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='ASTen',
    version='0.1.0',
    author='ASTen Team',
    description='A minimal PyTorch-like deep learning library',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    packages=['ASTen'],
    ext_modules=[CMakeExtension('ASTen._C')],
    cmdclass={'build_ext': CMakeBuild},
    install_requires=[
        'numpy>=1.19.0',
        'pybind11>=2.6.0',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: C',
        'Programming Language :: C++',
    ],
)