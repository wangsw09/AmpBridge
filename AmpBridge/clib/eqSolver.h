#ifndef __EQSOLVER__
#define __EQSOLVER__

double bisectSearch(double (*func)(double ), double lower, double upper, double tol);

struct bound _bisectSearch_findUpper1_(double (*func)(double ), double lower, double incre_unit);  // search upward from lower, double incre_unit each time, return a valid {lower, upper}
struct bound _bisectSearch_findLower1_(double (*func)(double ), double upper, double decre_unit);  // search downward from upper, double decre_unit each time, return a valid {lower, upper}


struct bound _bisectSearch_findUpper2_(double (*func)(double ), double lower, double stop);  // search upward from lower, upper bound < stop, return a valid {lower, upper}
struct bound _bisectSearch_findLower2_(double (*func)(double ), double upper, double stop);  // search downward from upper, lower bound > stop, return a valud {lower, upper}

int sign(double x);

struct bound{
    double lower;
    double upper;
};

#endif
