#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T>
static void tblis_scalv_ref(dim_t n, T alpha, T* restrict A, inc_t inc_A)
{
    if (inc_A == 1)
    {
        for (dim_t i = 0;i < n;i++)
        {
            A[i] *= alpha;
        }
    }
    else
    {
        for (dim_t i = 0;i < n;i++)
        {
            (*A) *= alpha;
            A += inc_A;
        }
    }
}

template <typename T>
static void tblis_scalv_int(T alpha, Matrix<T>& A)
{
    if (A.length() > 1)
    {
        tblis_scalv_ref(A.length(), alpha, A.data(), A.row_stride());
    }
    else
    {
        tblis_scalv_ref(A.width(), alpha, A.data(), A.col_stride());
    }
}

template <typename T>
void tblis_scalv(dim_t n, T alpha, T* A, inc_t inc_A)
{
    if (alpha == 0)
    {
        tblis_zerov(n, A, inc_A);
    }
    else if (alpha == 1)
    {
        //do nothing
        return;
    }

    if (n == 0) return;

    tblis_scalv_ref(n, alpha, A, inc_A);
}

template <typename T>
void tblis_scalv(T alpha, Matrix<T>& A)
{
    if (alpha == 0)
    {
        tblis_zerov(A);
    }
    else if (alpha == 1)
    {
        //do nothing
        return;
    }

    Matrix<T> Av;

    ViewNoTranspose(A, Av);

    if (A.length() == 0 || A.width() == 0) return;

    tblis_scalv_int(alpha, A);
}

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void tblis_scalv(dim_t n, T alpha, T* A, inc_t inc_A); \
template void tblis_scalv(T alpha, Matrix<T>& A);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

}
}
