from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules_cscalar = [
        Extension("AmpBridge.cscalar.gaussian", sources=["AmpBridge/cscalar/gaussian.pyx"], libraries=["m"]),
        Extension("AmpBridge.cscalar.proximal", sources=["AmpBridge/cscalar/proximal.pyx"], libraries=["m"]),
        Extension("AmpBridge.cscalar.amp_mse", sources=["AmpBridge/cscalar/amp_mse.pyx"], libraries=["m"], include_dirs=["."]),
        Extension("AmpBridge.cscalar.wrapper", sources=["AmpBridge/cscalar/wrapper.pyx"], include_dirs=["."])
        ]

ext_modules_coptimization = [
        Extension("AmpBridge.coptimization.bridge_coord_desc",
            sources=["AmpBridge/coptimization/bridge_coord_desc.pyx"])
        ]

ext_modules_cblas = [
        Extension("AmpBridge.clinalg.cython_blas_wrapper",
            sources=["AmpBridge/clinalg/cython_blas_wrapper.pyx"])
        ]


setup(
        name = "cAccelerate",
        cmdclass = {"build_ext" : build_ext},
        ext_modules = ext_modules_cscalar + ext_modules_coptimization + ext_modules_cblas,
        script_args=["build_ext"],
        options={"build_ext" : {"inplace" : True, "force" : True}},
        )

