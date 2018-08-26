import numpy as np
import numpy.random as npr


numeric = [int, float]

class ddist(object):
    '''
    Class <ddist>: describe discrete distribution
    '''
    def __init__(self, support, prob):
        self.support = support
        self.probability = prob
        self.dist = {support[i] : prob[i] for i in xrange(len(support))}

    def expectation(self, fun, **kwargs):
        expect = 0
        for x in self.dist:
            expect += fun(x, **kwargs) * self.dist[x]
        return expect

    def mean(self):
        ret = 0
        for x in self.dist:
            ret += x * self.dist[x]
        return ret

    def var(self):
        moment1 = 0
        moment2 = 0
        for x in self.dist:
            moment1 += x * self.dist[x]
            moment2 += x ** 2 * self.dist[x]
        return moment2 - moment1 ** 2

    def supp(self):
        return self.dist.keys()

    def prob(self):
        return self.dist.values()

    def items(self):
        return self.dist.items()

    def append(self, new_supp, new_prob):  # IMPORTANT: please add this function later, and replace the relevant part in the whole program later
        '''
        x is the new atom
        p is the new probability
        re-weight the old probability by (1 - p)
        return a new distributtion
        NOTE: later can re-define '+' for returning new dist; let add_atom() mutates the old distribution
        '''
        if s in self.dist:
            raise ValueError("The support is already in the distribution.")
        else:
            for s in self.dist.keys():
                self.dist[s] *= 1 - new_prob
            self.dist[new_supp] = new_prob
        
    def rm(self, s):
        '''
        delete a atom with support s and probability p
        '''
        p = self.dist[s]
        del self.dist[s]
        for k in self.dist:
            self.dist[k] /= 1.0 - p

    #def _sample(self):
    #    '''
    #    sample 1 observation from itself
    #    '''
    #    p_sumed = 0.0
    #    for s, p in self.dist.items():
    #        if p / (1.0 - p_sumed) >= 1:
    #            return s
    #        tmp = npr.binomial(1, p / (1.0 - p_sumed))
    #        if tmp == 1:
    #            return s
    #        else:
    #            p_sumed += p

    def sample(self, n):
        samp_arr = - np.ones(n)
        for i in xrange(n):
            tmp = np.nonzero(npr.multinomial(1, self.probability))[0][0]
            samp_arr[i] = self.support[tmp]
        return samp_arr
            

    def __str__(self):
        return str(self.dist.items())

    def convolution(self, dist2):
        pass

    def cdf(self):
        pass


