from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
# from Cython.Build import cythonize

ext_modules = [
        Extension("clib", sources=["clib.pyx"], libraries=["m"]),
        Extension("gaussian", sources=["gaussian.pyx"], libraries=["m"]),
        Extension("proximal", sources=["proximal.pyx"], libraries=["m"])
        ]

setup(
        name = "cAccelerate",
        cmdclass = {"build_ext" : build_ext},
        ext_modules = ext_modules
        )

