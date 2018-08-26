from math import sqrt, pi, exp
import sys
import numpy as np
import ctypes
from .tools import *
from .base_class import *
from ..cscalar import *


eta = np.vectorize(prox_Lq)
 
def eta_derivative(u, t, q=1, tol=1e-9):
    '''
    derivative of eta() w.r.t. u
    '''
    if q == 1:
        return (np.absolute(u) > t).astype(int)
    elif q == 2:
        if type(u) == np.ndarray:
            return np.repeat(1.0 / (1.0 + 2.0 * t), len(u))
        else:
            return 1.0 / (1.0 + 2.0 * t)
    elif q == 1.5:
        return 1.0 - t / np.sqrt(t ** 2 + 1.77777777778 * np.absolute(u))
    elif q < 2:
        tmp = np.absolute(eta(u, t, q, tol)) ** (2.0 - q)
        return tmp / (tmp + t * q * (q - 1.0))
    elif q > 2:
        tmp = np.absolute(eta(u, t, q, tol)) ** (q - 2.0)
        return 1 / (1 + t * q * (q - 1.0) * tmp)

def eta_inv(v, t, q):
    '''
    Compute u in v = eta(u, t, q), that is, the inverse of eta function.
    '''
    if q == 1:
        raise ValueError("q must be larger than 1.")
    return v + q * t * np.absolute(v) ** (q - 1.0) * np.sign(v)

def prox_sortedL1(y_raw, lam):
    '''
    this function calculate the proximal mapping function
    for the sorted-L1 norm
     -- <y_raw> is an array which may not be sorted
     -- <lam> is a vector of tunings in decreasing order, len(y) = len(lam)
     -- the notation follows that in (2.3) in the paper [BBSSC15]
     -- the algorithm is implemented based on [Algorithm 4], the [FastProxSL1] in [BBSSC15]
     -- seems this algorithm has some typos
     ** error handling
     ** broadcast <lam> when the length of it is smaller than that of <y>
    '''
    
    # find optimal group levels
    # notice that reduce the index for <t> and <k> by 1 to fit for python
    # replace <n> by <p> to fit notation of (2.3)
    
    t = -1
    p = len(y_raw)
    
    # error handle
    if type(lam) in numeric or len(lam) == 1:  # should process the len(lam) == 1 separately
        lam = np.repeat(lam, p)
    elif len(lam) < p:
        raise ValueError('The length of <lam> is not compatible; either \"len(lam) == 1\" or "len(lam) == len(y_raw)";')
    
    ## sort <y>
    y = sorted([[a, i] for i, a in enumerate(np.absolute(y_raw))], key=lambda ell: ell[0], reverse=True)
    x = -3.14159 * np.ones(p)

    ijsw = []  # the stack to store the tuple (i, j, s, w)
    for k in xrange(p):
        t += 1
        i = k
        j = k
        s = y[i][0] - lam[i]
        w = max(s, 0)
        
        ijsw.append([i, j, s, w])
        
        while t > 0 and ijsw[t - 1][3] <= ijsw[t][3]:
            ijsw[t - 1][1] = ijsw[t][1]
            ijsw[t - 1][2] += ijsw[t][2]
            ijsw[t - 1][3] = max(ijsw[t - 1][2] / (ijsw[t][1] - ijsw[t - 1][0] + 1.0), 0.0)
#            ijsw[t - 1][3] = max((ijsw[t - 1][1] - ijsw[t - 1][0] + 1) / (ijsw[t][1] - ijsw[t - 1][0] + 1) * ijsw[t - 1][2]
#                                 + (ijsw[t][1] - ijsw[t][0] + 1) / (ijsw[t][1] - ijsw[t - 1][0] + 1) * ijsw[t][2], 0)
            ijsw.pop()
            t -= 1
    
    # set entries in <x> for each block
    for l in xrange(t + 1):
        for k in xrange(ijsw[l][0], ijsw[l][1]+1):
            x[y[k][1]] = ijsw[l][3]
            
    return x * np.sign(y_raw)







def prox_sortedL1(y_raw, lam):
    '''
    this function calculate the proximal mapping function
    for the sorted-L1 norm
     -- <y_raw> is an array which may not be sorted
     -- <lam> is a vector of tunings in decreasing order, len(y) = len(lam)
     -- the notation follows that in (2.3) in the paper [BBSSC15]
     -- the algorithm is implemented based on [Algorithm 4], the [FastProxSL1] in [BBSSC15]
     -- seems this algorithm has some typos
     ** error handling
     ** broadcast <lam> when the length of it is smaller than that of <y>
    '''
    
    # find optimal group levels
    # notice that reduce the index for <t> and <k> by 1 to fit for python
    # replace <n> by <p> to fit notation of (2.3)
    
    t = -1
    p = len(y_raw)
    
    if type(lam) in numeric or len(lam) == 1:  # should process the len(lam) == 1 separately
        lam = np.repeat(lam, p)
    elif len(lam) < p:
        raise ValueError('The length of <lam> is not compatible; either \"len(lam) == 1\" or "len(lam) == len(y_raw)";')
    
    ## sort <y>
    y = sorted([[a, i] for i, a in enumerate(np.absolute(y_raw))], key=lambda ell: ell[0], reverse=True)
    x = -3.14159 * np.ones(p)

    ijsw = []  # the stack to store the tuple (i, j, s, w)
    for k in xrange(p):
        t += 1
        i = k
        j = k
        s = y[i][0] - lam[i]
        w = max(s, 0)
        
        ijsw.append([i, j, s, w])
        
        while t > 0 and ijsw[t - 1][3] <= ijsw[t][3]:
            ijsw[t - 1][1] = ijsw[t][1]
            ijsw[t - 1][2] += ijsw[t][2]
            ijsw[t - 1][3] = max(ijsw[t - 1][2] / (ijsw[t][1] - ijsw[t - 1][0] + 1.0), 0.0)
#            ijsw[t - 1][3] = max((ijsw[t - 1][1] - ijsw[t - 1][0] + 1) / (ijsw[t][1] - ijsw[t - 1][0] + 1) * ijsw[t - 1][2]
#                                 + (ijsw[t][1] - ijsw[t][0] + 1) / (ijsw[t][1] - ijsw[t - 1][0] + 1) * ijsw[t][2], 0)
            ijsw.pop()
            t -= 1
    
    # set entries in <x> for each block
    for l in xrange(t + 1):
        for k in xrange(ijsw[l][0], ijsw[l][1]+1):
            x[y[k][1]] = ijsw[l][3]
            
    return x * np.sign(y_raw)


