#include "eqSolver.h"

// bisectSearch() will do bisection search to find the root of function func()
// This function will not do pre-error checking;
// user must guarantee: lower < upper; sign of func at lower and upper differs.

double bisectSearch(double (*func)(double ), double lower, double upper, double tol)
{
    double mid;
    int sign_lower = sign(func(lower));
    int sign_upper = sign(func(upper));

    while (upper - lower > tol)
    {
        mid = (lower + upper) / 2;
        if (sign(func(mid)) == sign_upper)
            upper = mid;
        else
            lower = mid;
    }

    return (upper + lower) / 2;
}


// search upward from lower, double incre_unit each time, return a valid {lower, upper}
struct bound _bisectSearch_findUpper1_(double (*func)(double ), double lower, double incre_unit)
{
    double upper = lower + incre_unit;
    struct bound ret_bound;

    while (sign(func(lower)) == sign(func(upper)))
    {
        incre_unit *= 2;
        lower = upper;
        upper += incre_unit;
    }

    ret_bound.lower = lower;
    ret_bound.upper = upper;
    return ret_bound;
}

// search downward from upper, double decre_unit each time, return a valid {lower, upper}

struct bound _bisectSearch_findLower1_(double (*func)(double ), double upper, double decre_unit)
{
    double lower = upper - decre_unit;
    struct bound ret_bound;

    while (sign(func(lower)) == sign(func(upper)))
    {
        decre_unit *= 2;
        upper = lower;
        lower -= decre_unit;
    }

    ret_bound.lower = lower;
    ret_bound.upper = upper;
    return ret_bound;
}


 // search upward from lower, upper bound < stop, return a valid {lower, upper}

struct bound _bisectSearch_findUpper2_(double (*func)(double ), double lower, double stop)
// search downward from upper, lower bound > stop, return a valud {lower, upper}
{
    double gap = (stop - lower) / 2.0;
    double upper = lower + gap;
    struct bound ret_bound;

    while (sign(func(lower)) == sign(func(upper)))
    {
        gap /= 2;
        lower = upper;
        upper += gap;
    }

    ret_bound.lower = lower;
    ret_bound.upper = upper;
    return ret_bound;
}

struct bound _bisectSearch_findLower2_(double (*func)(double ), double upper, double stop)
{
    double gap = (upper - stop) / 2.0;
    double lower = upper - gap;
    struct bound ret_bound;

    while (sign(func(lower)) == sign(func(upper)))
    {
        gap /= 2;
        upper = lower;
        lower -= gap;
    }

    ret_bound.lower = lower;
    ret_bound.upper = upper;
    return ret_bound;
}

int sign(double x)
{
    return (x > 0) - (x < 0);
}
