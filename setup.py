from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
        Extension("AmpBridge.cscalar.clib", sources=["AmpBridge/cscalar/clib.pyx"], libraries=["m"]),
        Extension("AmpBridge.cscalar.gaussian", sources=["AmpBridge/cscalar/gaussian.pyx"], libraries=["m"]),
        Extension("AmpBridge.cscalar.proximal", sources=["AmpBridge/cscalar/proximal.pyx"], libraries=["m"]),
        Extension("AmpBridge.cscalar.amp_mse", sources=["AmpBridge/cscalar/amp_mse.pyx"], libraries=["m"], include_dirs=["."]),
        Extension("AmpBridge.cscalar.wrapper", sources=["AmpBridge/cscalar/wrapper.pyx"], include_dirs=["."])
        ]

setup(
        name = "cAccelerate",
        cmdclass = {"build_ext" : build_ext},
        ext_modules = ext_modules,
        script_args=["build_ext"],
        options={"build_ext" : {"inplace" : True, "force" : True}},
        )

