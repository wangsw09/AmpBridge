#include <math.h>
#include <stdio.h>
#include "cProxOperator.h"

// for our case, use secant method, in stead of false positive method should be very fast.
// false positive method will not work (but why work in our case?) because the interval
// does not shrink to 0.

double cprox_Lq(double x, double lambda, double q, double tol)
{
    if (x < 0)
        return (- cprox_Lq(-x, lambda, q, tol));

    if (q == 1)
        return (x - lambda) * (x > lambda);
    else if (q == 2)
        return x / (1.0 + 2.0 * lambda);
    else if (q > 1)
    {
        if (x < tol)
            return 0;

        double z1 = 0.0;
        double z2 = x;
        double f1, f2;

        double s;

        while (fabs(z2 - z1) > tol)
        {
            f1 = z1 + lambda * q * pow(z1, q - 1.0);
            f2 = z2 + lambda * q * pow(z2, q - 1.0);

            
            s = (z1 * f2 - z2 * f1 + (z2 - z1) * x) / (f2 - f1);
            z1 = z2;
            z2 = (s > 0.0) ? s : 0.0;
//            printf("(z1, z2): (%f, %f)  s: %f\n", z1, z2, s);
//            if (s + lambda * q * pow(s, q - 1) > x)
//                u = s;
//            else
//                l = s;
        }
        return z2;
    }

    else if (q == 0)
        return x * (x > lambda);
    else
        if (x < pow(lambda, 1.0 / (2.0 - q)) * pow(2.0 * (1.0 - q), 1.0 / (2.0 - q)) * (2.0 - q) / (2.0 - 2.0 * q))
            return 0;
        else
        {
            return 0;
        }
}



