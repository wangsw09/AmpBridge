## Package AmpBridge

This is a package from my research with Haolei Weng and Professor Arian Maleki.
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




################################################
##     (Expected) Structure of this module
################################################

MordernLM
    |
    |----- lib
    |          -- 
    |          -- prox_func.py
    |          -- eqsolver.py
    |
    |----- data_gen
    |          -- gen_lm.py
    |      
    |----- linear_model.py
    |          -- AMP
    |          -- Bridge Reg
    |          -- SLOPE
    |        
    |----- * AMP_func.py
    |          
    |----- AMPtheory
    |          -- AMPbasic.py
    |              --- state_evol
    |              --- calib
    |              --- optm_alpha
    |              --- optm_tau
    |              --- optm_lambda
    |          -- mseAnalysis.py
    |              --- large_noise
    |              --- low_epsilon
    |              --- low_noise
    |              --- large_delta
    |----- clib:
    |           -- cprox.py
    |           -- ceqsolver.py
    |
    |
    |----- test
    |
    |
    |
    |
    |


