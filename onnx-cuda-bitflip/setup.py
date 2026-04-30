from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import sys
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
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        # CMake also auto-discovers ../onnxruntime-dev when these are not set.
        ort_include_dir = os.environ.get('ONNXRUNTIME_INCLUDE_DIR', '')
        ort_library = os.environ.get('ONNXRUNTIME_LIBRARY', '')
        ort_lib_dir = os.environ.get('ONNXRUNTIME_LIB_DIR', '')

        if ort_include_dir:
            cmake_args.append(f'-DONNXRUNTIME_INCLUDE_DIR={ort_include_dir}')
        if ort_library:
            cmake_args.append(f'-DONNXRUNTIME_LIBRARY={ort_library}')
        if ort_lib_dir:
            cmake_args.append(f'-DONNXRUNTIME_LIBRARY={os.path.join(ort_lib_dir, "libonnxruntime.so")}')

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version()
        )
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, 
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, 
                              cwd=self.build_temp)

setup(
    name='onnx_bitflip',
    version='0.1',
    author='Claude',
    author_email='example@example.com',
    description='ONNX BitFlip custom operator',
    long_description='',
    packages=find_packages('python'),
    package_dir={'': 'python'},
    ext_modules=[CMakeExtension('onnx_bitflip')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'onnxruntime-gpu',
    ],
)
