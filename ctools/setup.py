import numpy
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            name="puzzler.ctools",
            sources=["puzzler_ctools.c", "nearest_point.cxx"],
            include_dirs=[numpy.get_include()],
        )
    ]
)
