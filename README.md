## Package AmpBridge

1. This package provides algorithms for the following topics:
   * AMP (approximated message passing) with Lq proximal operator
   * Linear regression with Lq penalty, for any q >= 1
   * Related functions to the above two topics (such as tuning)
   * Numerical calculation about several quantities on penalized linear regression, such as MSE, optimal tuned MSE, optimal tuning parameter, etc.

2. By using Cython to accelerate critical components, the package provides a fast solutions to the above topics.
3. This work originates from my research with Haolei Weng and Professor Arian Maleki at Columbia University.

## Index
1. Introduction
2. Examples
3. Details
4. [Update](#update)

## Some Examples

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

## Details
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


