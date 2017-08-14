## Package AmpBridge

This is a package from my research with Haolei Weng and Professor Arian Maleki.


Some examples of usage.

1. Calculate the theoretical mse-related quantity

```python
import AmpBridge as ab

eps = 0.3
delta = 0.7
sigma = 0.5
nonzero_dist = ab.ddist([1], [1])

ath = ab.amp_theory(eps=eps, delta=delta, sigma=sigma, nonzero_dist=nonzero_dist)

# optimal tuning
q = 1.2
alpha = ath.alpha_optimal(q)
tau = ath.tau_of_alpha(alpha, q)
mse = ath.mse(alpha, tau, q)
```

2. Calculate bridge regression and sample "MSE"
```python
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
```


It implementes the following content

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




