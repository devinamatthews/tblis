#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T>
static void tblis_normfv_ref(dim_t n, const T* restrict A, inc_t inc_A, T& restrict norm)
{
    norm = T();

    if (inc_A == 1)
    {
        for (dim_t i = 0;i < n;i++)
        {
            norm += norm2(A[i]);
        }
    }
    else
    {
        for (dim_t i = 0;i < n;i++)
        {
            norm += norm2(*A);
            A += inc_A;
        }
    }

    norm = sqrt(real(norm));
}

template <typename T>
static void tblis_normfv_int(const Matrix<T>& A, T& norm)
{
    if (A.length() > 1)
    {
        tblis_normfv_ref(A.length(), A.data(), A.row_stride(), norm);
    }
    else
    {
        tblis_normfv_ref(A.width(), A.data(), A.col_stride(), norm);
    }
}

template <typename T>
void tblis_normfv(dim_t n, const T* A, inc_t inc_A, T& norm)
{
    if (n == 0) return;
    tblis_normfv_ref(n, A, inc_A, norm);
}

template <typename T>
T tblis_normfv(dim_t n, const T* A, inc_t inc_A)
{
    T norm;
    tblis_normfv(n, A, inc_A, norm);
    return norm;
}

template <typename T>
void tblis_normfv(const Matrix<T>& A, T& norm)
{
    Matrix<T> Av;

    ViewNoTranspose(const_cast<Matrix<T>&>(A), Av);

    if (A.length() == 0 || A.width() == 0) return;

    tblis_normfv_int(A, norm);
}

template <typename T>
T tblis_normfv(const Matrix<T>& A)
{
    T norm;
    tblis_normfv(A, norm);
    return norm;
}

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void tblis_normfv(dim_t n, const T* A, inc_t inc_A, T& norm); \
template    T tblis_normfv(dim_t n, const T* A, inc_t inc_A); \
template void tblis_normfv(const Matrix<T>& A, T& norm); \
template    T tblis_normfv(const Matrix<T>& A);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

}
}
