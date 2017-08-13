#include <stdio.h>
#include "eqSolver.h"

double func1(double x)
{
    return (x - 1) * x * x;
}

int main()
{
    printf("%d\n", sign(20));
    printf("%d\n", sign(-1.5));
    printf("%f\n", bisectSearch(&func1, -3, 5, 0.00000001));
    return 0;
}
