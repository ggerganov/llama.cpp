import os
import re
import platform
import sys
import sysconfig
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

curr_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(curr_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        if cmake_version < '3.15':
            raise RuntimeError("CMake >= 3.15 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DKOMPUTE_OPT_BUILD_PYTHON=ON',
                      '-DKOMPUTE_OPT_LOG_LEVEL=Off',
                      '-DKOMPUTE_OPT_USE_SPDLOG=Off',
                      '-DKOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK=ON'
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DPYTHON_INCLUDE_DIR=' + sysconfig.get_path('include'),
                      '-DPYTHON_LIBRARY=' + sysconfig.get_path('stdlib'),
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        env = os.environ.copy()
        oldCxxFlags = env.get('CXXFLAGS', '')
        env['CXXFLAGS'] = f'{oldCxxFlags} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'

        if platform.system() == "Windows":
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}']
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            env['CXXFLAGS'] += ' -fPIC'
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j']
            # Optional environment variable to limit the number of parallel jobs for GitHub actions to reduce RAM usage
            if 'KOMPUTE_PYTHON_NUM_PARALLEL_THREADS' in env:
                build_args += env['KOMPUTE_PYTHON_NUM_PARALLEL_THREADS']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='kp',
    version='0.8.1',
    author='Alejandro Saucedo',
    description='Kompute: Blazing fast, mobile-enabled, asynchronous, and optimized for advanced GPU processing usecases.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    ext_modules=[CMakeExtension('kp')],
    install_requires=[
        "numpy<2.0.0"
    ],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    include_package_data=True,
)
