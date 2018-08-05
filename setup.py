from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
        Extension("AmpBridge.cplib.clib", sources=["AmpBridge/cplib/clib.pyx"], libraries=["m"]),
        Extension("AmpBridge.cplib.gaussian", sources=["AmpBridge/cplib/gaussian.pyx"], libraries=["m"]),
        Extension("AmpBridge.cplib.proximal", sources=["AmpBridge/cplib/proximal.pyx"], libraries=["m"]),
        Extension("AmpBridge.cplib.amp_mse", sources=["AmpBridge/cplib/amp_mse.pyx"], libraries=["m"], include_dirs=["."]),
        Extension("AmpBridge.cplib.wrapper", sources=["AmpBridge/cplib/wrapper.pyx"], include_dirs=["."])
        ]

setup(
        name = "cAccelerate",
        cmdclass = {"build_ext" : build_ext},
        ext_modules = ext_modules,
        script_args=["build_ext"],
        options={"build_ext" : {"inplace" : True, "force" : True}},
        )

