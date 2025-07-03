#!/usr/bin/env python
"""
最终修正版的 setup.py 脚本
- 自动检测 Windows + Conda 环境中的 GSL 路径
- 使用正确的 build_ext 命令类
"""
import os
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
# --- 关键修正：从 Cython.Distutils 导入 build_ext ---
from Cython.Distutils import build_ext
import numpy

# --- 自动检测Conda环境路径 (这部分是正确的，予以保留) ---
if 'CONDA_PREFIX' in os.environ:
    conda_env_path = os.environ['CONDA_PREFIX']
    gsl_include_path = os.path.join(conda_env_path, 'Library', 'include')
    gsl_lib_path = os.path.join(conda_env_path, 'Library', 'lib')
else:
    raise EnvironmentError("错误：看起来您不在一个激活的Conda环境中。请先 'conda activate gcp_pot_env'")

if not os.path.isdir(gsl_include_path):
    raise FileNotFoundError(f"错误：在Conda环境路径中找不到GSL的include目录: {gsl_include_path}")
if not os.path.isdir(gsl_lib_path):
    raise FileNotFoundError(f"错误：在Conda环境路径中找不到GSL的lib目录: {gsl_lib_path}")
# --- 自动检测代码结束 ---

# --- 平台适配代码 (这部分是正确的，予以保留) ---
if sys.platform == 'win32':
    extra_compile_args = []
    extra_link_args = []
else:
    extra_compile_args = ["-std=c++11"]
    extra_link_args = ["-std=c++11"]
# --- 平台适配代码结束 ---

extensions = [
    Extension('series',
              sources=['series.pyx', 'bessel.c'],
              include_dirs=[
                  numpy.get_include(),
                  gsl_include_path
              ],
              library_dirs=[
                  gsl_lib_path
              ],
              libraries=['gsl', 'gslcblas'],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args
              )
]

setup(
    name='functions',
    author='kruskallin',
    author_email='kruskallin@tamu.edu',
    # --- 关键修正：将 cmdclass 指向正确的 build_ext ---
    cmdclass={'build_ext': build_ext},
    # --- ext_modules 的用法是正确的，予以保留 ---
    ext_modules=cythonize(extensions),
    install_requires=[
        "numpy >= 1.13",
    ],
    zip_safe=False,
)