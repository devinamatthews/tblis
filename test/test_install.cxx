#include <tblis/tblis.h>

using namespace tblis;

int main()
{
    varray<double> A({10, 4, 3});
    varray<double> B({3, 4, 6, 7});
    varray<double> C({7, 10, 6});

    mult(1.0, A, "abc", B, "cbde", 0.0, C, "ead");
}
