#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T>
static void tblis_zerov_ref(dim_t n, T* restrict A, inc_t inc_A)
{
    if (inc_A == 1)
    {
        for (dim_t i = 0;i < n;i++)
        {
            A[i] = T();
        }
    }
    else
    {
        for (dim_t i = 0;i < n;i++)
        {
            (*A) = T();
            A += inc_A;
        }
    }
}

template <typename T>
static void tblis_zerov_int(Matrix<T>& A)
{
    if (A.length() > 1)
    {
        tblis_zerov_ref(A.length(), A.data(), A.row_stride());
    }
    else
    {
        tblis_zerov_ref(A.width(), A.data(), A.col_stride());
    }
}

template <typename T>
void tblis_zerov(dim_t n, T* A, inc_t inc_A)
{
    if (n == 0) return;
    tblis_zerov_ref(n, A, inc_A);
}

template <typename T>
void tblis_zerov(Matrix<T>& A)
{
    Matrix<T> Av;
    ViewNoTranspose(A, Av);
    if (A.length() == 0 || A.width() == 0) return;
    tblis_zerov_int(A);
}

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void tblis_zerov(dim_t n, T* A, inc_t inc_A); \
template void tblis_zerov(Matrix<T>& A);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

}
}
