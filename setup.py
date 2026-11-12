from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [Extension("sum_squares_cython", ["sum_squares_cython.pyx"])]
setup(
    name="numpy-lab-cython",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": 3}),
)
