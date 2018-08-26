import sys
import os.path

import numpy as np


def bisect_search(fun, lower=None, upper=None, bound=None, unit=None,
                  lower_sign=None, upper_sign=None,
                  accuracy=1e-10, **kwargs):
    '''
    fun(lower) and fun(upper) must have different signs
    * Re-write this part in Cython.
    #
    '''
    if (lower is None or upper is None) and bound is None and unit is None:
        raise ValueError('when lower or upper is missing, an estimated (upper/lower) bound or trial unit must be provided')
    if lower is None and upper is None:
        raise ValueError('at least one of lower and upper should be provided')
    if lower is not None and upper is not None and np.sign(fun(lower, **kwargs)) == np.sign(fun(upper, **kwargs)):
        raise ValueError('function has same signs on lower and upper bound')
    
    if lower is None:
        if bound is not None:
            if bound >= upper:
                raise ValueError('bound must be lower bound when lower is missing')

            distance = (upper - bound) / 2.0
            lower = upper - distance
            lower_sign = np.sign(fun(lower, **kwargs))
            upper_sign = np.sign(fun(upper, **kwargs))
            
            while upper_sign == lower_sign:
                distance /= 2.0
                lower -= distance
                lower_sign = np.sign(fun(lower, **kwargs))
            upper = lower + distance
        else:
            gamma = 2
            distance = unit
            lower = upper - distance
            lower_sign = np.sign(fun(lower, **kwargs))
            upper_sign = np.sign(fun(upper, **kwargs))
            while upper_sign == lower_sign:
                distance *= gamma
                lower -= distance
                lower_sign = np.sign(fun(lower, **kwargs))
            upper = lower + distance
    
    if upper is None:
        if bound is not None:
            if bound <= lower:
                raise ValueError('bound must be upper bound when upper is missing')
            
            distance = (bound - lower) / 2.0
            upper = lower + distance
            if lower_sign is None:
                lower_sign = np.sign(fun(lower, **kwargs))
            if upper_sign is None:
                upper_sign = np.sign(fun(upper, **kwargs))
            
            while lower_sign == upper_sign:
                distance /= 2.0
                upper += distance
                upper_sign = np.sign(fun(upper, **kwargs))
            lower = upper - distance
        else:
            gamma = 2
            distance = unit
            upper = lower + distance
            if lower_sign is None:
                lower_sign = np.sign(fun(lower, **kwargs))
            if upper_sign is None:
                upper_sign = np.sign(fun(upper, **kwargs))
            while lower_sign == upper_sign:
                distance *= gamma
                upper += distance
                upper_sign = np.sign(fun(upper, **kwargs))
            lower = upper - distance
    
    if lower_sign is None:
        lower_sign = np.sign(fun(lower, **kwargs))
    if upper_sign is None:
        upper_sign = np.sign(fun(upper, **kwargs))
    
    while (upper - lower) > accuracy:
        middle = (lower + upper) / 2.0
        middle_sign = np.sign(fun(middle, **kwargs))
        if middle_sign == upper_sign:
            upper = middle
        else:
            lower = middle
    return (lower + upper) / 2.0



def normalize(z, type):
    '''
     -- this function returns the normalized vector or matrix
     -- if <type> = 'c', then just center <z>
     -- if <type> = 'cs', then center and scale <z>
    '''
    if type == 'c':
        return z - np.mean(z)
    elif type == 'cs':
        n, p = z.shape
        return (z - np.mean(z, 0)) / np.sqrt(np.sum(z ** 2.0, 0) - n * np.mean(z, 0) ** 2.0)
    else:
        raise ValueError('please correctly specify <type>')
    
def multi_test(pvals, q, method):
    '''
     -- this function does multiplie testing, with FDR or FWER controlled
     Arguments:
     -- <pvals> is an array of p-values from a multiple testing
     -- <q> is the level to be controlled
     -- <method> specifies the method to use:
          'Bonf': Bonferroni method, FWER is controlled
          'BH': Benjamini-Hochberg method, FDR is controlled
     Returns:
     -- return a vector with True or False entries, with True means reject and False means not-reject
    '''
    p = len(pvals)
    if method == 'Bonf':
        return (pvals <= q / float(p))
    elif method == 'BH':
        sorted_pvals = sorted([(pval, i) for i, pval in enumerate(pvals)], key = lambda a: a[0])
        k_star = p  # the first k_star hypotheses in the sorted list will be rejected
        for i in xrange(p, 0, -1):
            if sorted_pvals[i - 1][0] > i * q / p:
                k_star -= 1
            else:
                break
        ret = np.zeros(p, dtype=bool)
        for i in xrange(k_star):
            ret[sorted_pvals[i][1]] = True
        return ret

def fdp(theta_estimate, theta_true): # why not combine the following two functions into one tpp_fdp(); return a tuple (tpp, fdp)
    if len(theta_estimate) != len(theta_true):
        raise ValueError('the length of the two vectors are not the same')
    else:
        return np.sum((theta_estimate != 0) * (theta_true == 0)) / max(float(np.sum(theta_estimate != 0)), 1)

def tpp(theta_estimate, theta_true):
    if len(theta_estimate) != len(theta_true):
        raise ValueError('the length of the two vectors are not the same')
    else:
        return np.sum((theta_estimate != 0) * (theta_true != 0)) / max(float(np.sum(theta_true != 0)), 1)


