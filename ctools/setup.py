import numpy
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            name="puzzbin",
            sources=["puzzler_ctools.cxx", "nearest_point.cxx"],
            depends=["nearest_point.h"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["/std:c++20"],
        )
    ]
)
