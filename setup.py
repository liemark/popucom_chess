# setup.py
# 用于编译 Cython 模块的 setup.py 文件

from setuptools import setup
from Cython.Build import cythonize
import numpy # 需要 numpy 的头文件

setup(
    ext_modules=cythonize(
        "popucom_mcts_cython_utils.pyx",
        compiler_directives={'language_level': "3", 'binding': True} # binding=True 提供更好的错误报告
    ),
    include_dirs=[numpy.get_include()] # 包含 NumPy 头文件，如果你的 Cython 代码使用 np
)
