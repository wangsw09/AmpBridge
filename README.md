# AmpBridge
Approximate message passing (AMP), state-evolution analysis, bridge regression,
Stein's unbiased risk estimator (SURE) of bridge regression.

Approximate message passing is an algorithm which has tractable and analyzable
performance under specific design (Gaussian typically). On the other hand, it connects to a big class of statistical machine learning models, making it possible to analyze the statistical properties of these models, including MSE, variable selection, etc.

This package provides functionalities about the following topics:
* Theoretical quantities about AMP state-evolutions, such as MSE, optimal tuning,
  optimal MSE, transformation between raw tuning and AMP tuning, etc;
* The 1st (2nd) order expansion of the optimal MSE under different asymptotics;
* Bridge regression with Lq penalty, where q >= 1 (coordinate descent implementation)
* Empirical MSE and transformation between raw tuning and AMP tuning under Gaussian
  design.

`Cython` is used to accelerate the core, upon which the whole package is then built.
Regarding the calculations involving matrix operations, `blas` is used to boost the
performance (through calling `scipy.linalg.cython_blas`).

The work is motivated by my research with Haolei Weng and Professor Arian Maleki at Columbia University. The relevant paper is "[Which bridge estimator is optimal for variable selection?](http://arxiv.org/abs/1705.08617)". 

For examples to use the package, please check the `Examples.ipynb` in the root dir.

## Installation
After `git clone` the repo to your local dir, simply do the following to compile the
`cython` part:
```bash
make all
```
For windows user, please directly call `python setup build_ext --inplace --compiler=<specify_your_c_compiler>` and replace `<specify_your_c_compiler>` parameter with your C compiler.
