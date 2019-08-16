#include <tblis/tblis.h>

using namespace tblis;

int main()
{
    tensor<double> A({10, 4, 3});
    tensor<double> B({3, 4, 6, 7});
    tensor<double> C({7, 10, 6});

    mult<double>(1.0, A, "abc", B, "cbde", 0.0, C, "ead");
}
