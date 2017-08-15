## Package AmpBridge

1. This package provides algorithms for the following topics:
 * AMP (approximated message passing) with Lq proximal operator
 * Linear regression with Lq penalty, for any q >= 1
 * Related functions to the above two topics (such as tuning)
 * Numerical calculation about several quantities on penalized linear regression, such as MSE, optimal tuned MSE, optimal tuning parameter, etc.

2. By using Cython to accelerate critical components, the package provides a fast solutions to the above topics.
3. This work originates from my research with Haolei Weng and Professor Arian Maleki at Columbia University.

## Some Examples

1. Calculate the theoretical mse-related quantity

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
alpha = ath.alpha_optimal(q)
tau = ath.tau_of_alpha(alpha, q)
mse = ath.mse(alpha, tau, q)

print("The optimal MSE for q=1.2 under the above parameter settings is {0}".format(mse))
```

2. Calculate bridge regression and sample "MSE"
```python
from __future__ import print_function
import AmpBridge as ab

p = 8000
delta = 0.7
eps = 0.3
sigma = 0.5
signl = ab.ddist([1], [1])

y, X, beta_true = ab.data_gen.linmod(p, delta, eps, sigma, signl)

lm = ab.linear_model(y, X)
q = 1.2
lam = 4.0

beta_hat = lm.bridge(lam, q)
mse_hat = np.sum((beta_hat - beta_true) ** 2.0) / p

print("The sample MSE for q=1.2 under the optimal tuning is {0}".format(mse_hat))
print("The above two values should be very close.")
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




