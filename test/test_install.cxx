#include <tblis/tblis.h>

using namespace tblis;
using namespace MArray;

int main()
{
    marray<double> A{10, 4, 3};
    marray<double> B{3, 4, 6, 7};
    marray<double> C{7, 10, 6};

    mult(1.0, A, "abc", B, "cbde", 0.0, C, "ead");
}
