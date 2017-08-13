from ctypes import *
import timeit
import time


rel_path = './'

def main():
    eqSolver = cdll.LoadLibrary(rel_path + 'eqSolver.so')
    CNUMFUNC = CFUNCTYPE(c_double, c_double)
    lline = '------------------------------'
    sline = '---------------'

    class CBOUND(Structure):
        _fields_ = [('lower', c_double), ('upper', c_double)]

    eqSolver._bisectSearch_findUpper1_.restype = CBOUND
    eqSolver._bisectSearch_findUpper1_.argtypes = [CNUMFUNC, c_double, c_double]

    eqSolver._bisectSearch_findLower1_.restype = CBOUND
    eqSolver._bisectSearch_findLower1_.argtypes = [CNUMFUNC, c_double, c_double]

    eqSolver._bisectSearch_findUpper2_.restype = CBOUND
    eqSolver._bisectSearch_findUpper2_.argtypes = [CNUMFUNC, c_double, c_double]

    eqSolver._bisectSearch_findLower2_.restype = CBOUND
    eqSolver._bisectSearch_findLower2_.argtypes = [CNUMFUNC, c_double, c_double]

    eqSolver.bisectSearch.restype = c_double
    eqSolver.bisectSearch.argtypes = [CNUMFUNC, c_double, c_double, c_double]

    eqSolver.sign.restype = c_int
    eqSolver.sign.argtypes = [c_double]

    f = lambda x: (x - 5.4) ** 5
    cf = CNUMFUNC(f)

    print lline
    print "Testing Library <eqSolver.so>"
    print lline

    print "Part 1: bisection Search Methods"
    print sline
    print "f(x) = (x - 5.4) ** 5; Root: 5.4"
    print sline

    bd = eqSolver._bisectSearch_findUpper1_(cf, -3.1, 1.2)
    print "Test _bisectSearch_findUpper1_(f, -3.1, 1.2): Return (%f, %f)" % (bd.lower, bd.upper)
    print sline

    bd = eqSolver._bisectSearch_findLower1_(cf, 10.9, 1.2)
    print "Test _bisectSearch_findLower1_(f, 10.9, 1.2): Return (%f, %f)" % (bd.lower, bd.upper)
    print sline

    bd = eqSolver._bisectSearch_findUpper2_(cf, -2.9, 8)
    print "Test _bisectSearch_findUpper2_(f, -2.9, 8): Return (%f, %f)" % (bd.lower, bd.upper)
    print sline

    bd = eqSolver._bisectSearch_findLower2_(cf, 9.8, 0)
    print "Test _bisectSearch_findLower2_(f, 9.8, 0): Return (%f, %f)" % (bd.lower, bd.upper)
    print sline

    rt = eqSolver.bisectSearch(cf, 0.3, 7.8, 1e-8)
    print "Test bisectSearch(f, 0.3, 7.8, 1e-8): Return %f" % (rt,)
    print sline

    print "Test sign(-5.3): Return %d" % (eqSolver.sign(-5.3),)
    print lline




    cproxf = cdll.LoadLibrary(rel_path + 'cProxOperator.so')
    cproxf.cprox_Lq.restype = c_double
    cproxf.cprox_Lq.argtypes = [c_double, c_double, c_double, c_double]

    print lline
    print "Part 2: proximal functions"
    print sline
    print "Test cprox_Lq(10, 4.5, 1.50, 1e-8): Return %f" % (cproxf.cprox_Lq(10, 4.5, 1.50, 1e-8),)
    start = time.time()

    x = 1
    lam = 4.5
    q = 1.5
    for i in xrange(500000):
        cproxf.cprox_Lq(x, lam, 1.5, 1e-8)
    end = time.time()
    print "timing cproxf.cprox_Lq(" + str(x) + ", " + str(lam) + ", " + str(q) + ", 1e-8)"
    print "output: %f" % (cproxf.cprox_Lq(x, lam, q, 1e-8),)
    print "time cost: %fs" % ((end - start) / 500000.0,)
    print sline
    print lline




if __name__ == '__main__':
    main()
