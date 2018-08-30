cdef np.ndarray[dtype=np.float64_t, ndim=1] cprox_sortedL1(np.ndarray[dtype=np.float64_t, ndim=1] x,
        np.ndarray[dtype=np.float64_t, ndim=1] lam):
    """
    this function calculate the proximal mapping function
    for the sorted-L1 norm
     -- <x> is an array which may not be sorted
     -- <lam> is a vector of tunings in decreasing order, len(x) = len(lam)
     -- the notation follows that in (2.3) in the paper [BBSSC15]
     -- the algorithm is implemented based on [Algorithm 4], the [FastProxSL1] in [BBSSC15]
     -- seems this algorithm has some typos
     ** error handling
     ** broadcast <lam> when the length of it is smaller than that of <y>
    """
    # find optimal group levels
    # notice that reduce the index for <t> and <k> by 1 to fit for python
    # replace <n> by <p> to fit notation of (2.3)
    
    cdef int t = -1
    cdef int p = x.shape[0]
    
    # sort <y>
    y = sorted([[a, i] for i, a in enumerate(np.absolute(x))], key=lambda ell: ell[0], reverse=True)
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
            ijsw.pop()
            t -= 1
    
    # set entries in <x> for each block
    for l in xrange(t + 1):
        for k in xrange(ijsw[l][0], ijsw[l][1]+1):
            x[y[k][1]] = ijsw[l][3]
            
    return x * np.sign(x)

