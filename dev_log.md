8/26/2018
Switch back to work on this AmpBridge project, I want to finish it as soon as possible. Maybe in a week.

Previously I have already optimized the scalar part of the code in Cython, specifically: rewrote the proximal operators, and the function in amp mse for calculating the optimal tuning and optimal MSE;

Some components which can be further improved in the cscalar part include:
1. The Gaussian part can be implemented using the erf() function; no sure which is faster
   * Fixed, erf is much more accurate and 2 times faster;
2. Use c integration to replcae the scipy integration;
3. Use a better solver for the proximal operator of lq norm;


The next steps involve the following three main stages:
1. Write c optimized code for the Lq regression part;
   * Wrote one version with coordinate descent;
2. Merge the preivous optimized Amp c code into the current package;
3. Merge the new Lq regression part of c code into the current package.

Things to pay attention to:
1. The calculation of XTX is not memory-friendly. We may want to directly calculate its multiplication with a vector;

The structure of the package:

AmpBridge/
  __init__.py
  linear_model.py       penalized linear reg, optimal tuning, mse, variable selection methods
  lib/                  supposed to be replaced by cscalar. to be removed.
    __init__.py
    base_class.py       contains discrete distribution class, not useful for now
    prox_func.py        except for lq, also contains SLOPE; should copy SLOPE before abandon;
    tools.py            bisect_search, multi-test, normalize
  cscalar/              cython for scalar function;
    __init__.py
    proximal.pxd
    proximal.pyx        proximal function, derivatives for L_q, q >= 1
    gaussian.pxd
    gaussian.pyx        gaussian CDF, PDF, moments, expectation (not good to use)
    amp_mse.pxd
    amp_mse.pyx         mse, optimal tuning, etc for amp Lq
    wrapper.pyx         provide interface for outside python to access proximal and amp_mse
    clib.pyx            old cython module; contains bridge optimizer -- copied before removal
  coptimization/        plan to implement optimization algo for bridge regression
    __init__.py
    acc_grad_desc.pyx   plan to implement accelerated gradient descent
  AMPtheory/
    __init__.py
    AMPbasic.py         class containing amp mse fns; re-think a better way to wrap it;
    mseAnalysis.py      expansion of optimal mse in diff scenes. re-think whether need this..

