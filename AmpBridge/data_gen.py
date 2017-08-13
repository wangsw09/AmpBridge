import numpy as np
import numpy.random as npr

def linmod(p, delta, epsilon, sigma, signal):
    '''
    Function <data_generate>: generate data for linear model
    argument:
        delta, p, epsilon, sigma: float
        signal: the nonzero part G of the coefficient distribution
    * possible future extension:
    * build a module data_generate, with each type a function
    * such as "linear", "blahblahblah"
    '''
    n    = int(p * delta)  
    k    = int(p * epsilon)
    X    = npr.normal(0, 1 / np.sqrt(n), (n, p))

    beta = np.zeros(p)
    non_zero_loc       = npr.choice(p, k, False)
    beta[non_zero_loc] = signal.sample(k)
    
    w    = sigma * npr.normal(0, 1, n)
    y    = np.dot(X, beta) + w
    return (y, X, beta)

