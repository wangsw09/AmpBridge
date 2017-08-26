# Package AmpBridge

## Index
1. [Introduction](#introduction)
2. [Examples](#examples)
3. [Details](#details)
4. [Update](#update)


## <a name="introduction"></a>Introduction
This package provides classes and functions about the following topics:
* bridge regression with Lq penalty, where q >= 1
* AMP (approximated message passing) with Lq proximal operator
* Related functions to the above two topics (such as tuning and MSE)
* Numerical calculation about several quantities on penalized linear regression, such as MSE, optimal tuned MSE, optimal tuning parameter, etc.
* Theoretical quantities related to bridge regression under normal design, including the optimal tuning etc.

This work originates from my research with Haolei Weng and Professor Arian Maleki at Columbia University. The relevant paper is "[Which bridge estimator is optimal for variable selection?](http://arxiv.org/abs/1705.08617)". However this package provides codes for general usage related to bridge regression, AMP and theoretical calculation about tuning and MSE in AMP/bridge regression theories.

The package had a first version, which is fully written in Python. Currently I am re-implementing the package, mainly changing some structures, accelerating core parts through Cython and optimizing the algorithms. This updating is in progress. Please check [update](#update) for more details. The [examples](#examples) are compatible with the most up-to-date codes, which is something you would like to start with if you are using the code for the first time.


## <a name="examples"></a>Some Examples

1. Calculate the theoretical MSE-related quantity

```python
from __future__ import print_function
import AmpBridge as ab

# set basic parameters
eps = 0.3
delta = 0.7
sigma = 0.5
nonzero_dist = ab.ddist([1], [1])

# construct the amp_theory class object
ath = ab.amp_theory(eps=eps, delta=delta, sigma=sigma, nonzero_dist=nonzero_dist)

# optimal tuning for Lq penalty with q=1.2
q = 1.2
alpha = ath.alpha_optimal(q)  # 1.096
tau = ath.tau_of_alpha(alpha, q)  # 0.740
mse = ath.mse(alpha, tau, q)  # 0.208
lam = ath.lambda_of_alpha(alpha, q) # 0.44

print("The optimal MSE for q=1.2 under the above parameter settings is {0}".format(mse))
```

2. Calculate bridge regression and sample "MSE"
```python
from __future__ import print_function
import AmpBridge as ab
import numpy as np

p = 8000
delta = 0.7
eps = 0.3
sigma = 0.5
signl = ab.ddist([1], [1])

y, X, beta_true = ab.data_gen.linmod(p, delta, eps, sigma, signl)

brg = ab.bridge(q = 1.2)
q = 1.2

lam = 0.44
lam_arr = np.linspace(1, 0.1, 10)

beta = brg.fit(X, y, lam)  # bridge regression on single value of lam
beta_arr = brg.fit(X, y, lam_arr)  # bridge regression on array values of lam

fm = brg.fmse(beta, lam, X, y)  # fake MSE on single value of lam
fm = brg.fmse(beta_arr, lam_arr, X, y)  # fake MSE on array values of lam

m = brg.mse(beta, beta_true)  # MSE on single value of lam
m = brg.mse(beta_arr, beta_true)  # MSE on array values of lam

db = brg.debias(beta, lam, X, y)  # debiased version of single value of lam
db_arr = brg.debias(beta_arr, lam_arr, X, y)  # debiased version of array values of lam.

lam_opt = brg.auto_tune(X, y, 11)  # find the optimal tuning
```

## <a name="details"></a>Details
Package **AmpBridge** implements the follwoing contents

* proximal operator
  * proximal operator for Lq norm;
  * derivative and partial derivative of the above proximal operator;

* Approximate message passing (AMP) algorithm
  * AMP for Lq bridge regression with q>=1. (will do 0 <= q < 1 in the future);
  * Optimal tuning of AMP, based on reference.

* Theoretical results involved in AMP, including
  * state evolution
  * calibration equation
  * optimal tuning
  * etc...

* Bridge regression
  * Bridge regression for Lq with q >= 1;
  * Debiasing of bridge regression for i,i,d, Gaussian design;
  * Optimal tuning of bridge regression (to be finished);
  * Variable selection through bridge regression;
  * Variable Selection through debiased bridge regression;

* Slope

Future work:
* Other linear regression methods developed in recent years, including SCAD, adaptive Lasso, etc.

## <a name="update"></a>Recent Update
* Separate the functions related to bridge regression to a separate class <bridge>. In detail, this class contains the following functions
  * `fit()`: fit bridge regression.
  * `fmse()`: fake MSE. The tau ** 2 in state evolution.
  * `mse()`: MSE.
  * `debias()`: return the debiased version of the fitted estimator.
  * `auto_tune()`: provide optimal tuning for the bridge regression.
  * `__preprocess()`
  * `__cfit()`


