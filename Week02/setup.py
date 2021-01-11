from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

compile_flags = ['-std=c++11']

module = Extension('target_mean_cy',
                   ['target_mean_cy.pyx'],
                   language='c++',
                   include_dirs=[numpy.get_include()], # This helps to create numpy
                   extra_compile_args=compile_flags)

setup(
    name='target_mean_cy',
    ext_modules=cythonize(module),
    gdb_debug=True # This is extremely dangerous; Set it to False in production.
)
