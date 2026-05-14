import numpy
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            name="puzzler.ctools",
            sources=["puzzler_ctools.cxx", "nearest_point.cxx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["/std:c++20"],
        )
    ]
)
