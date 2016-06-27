#ifndef _TBLIS_SUMV_HPP_
#define _TBLIS_SUMV_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T>
static void tblis_sumv_ref(bool conj_A, dim_t n,
                           const T* restrict A, inc_t inc_A, T& restrict sum)
{
    sum = T();

    if (inc_A == 1)
    {
        for (dim_t i = 0;i < n;i++)
        {
            sum += A[i];
        }
    }
    else
    {
        for (dim_t i = 0;i < n;i++)
        {
            sum += (*A);
            A += inc_A;
        }
    }

    if (conj_A) sum = conj(sum);
}

template <typename T>
static void tblis_sumv_int(const Matrix<T>& A, T& sum)
{
    if (A.length() > 1)
    {
        tblis_sumv_ref(A.is_conjugated(), A.length(), A.data(), A.row_stride(), sum);
    }
    else
    {
        tblis_sumv_ref(A.is_conjugated(), A.width(), A.data(), A.col_stride(), sum);
    }
}

template <typename T>
void tblis_sumv(dim_t n, const T* A, inc_t inc_A, T& sum)
{
    if (n == 0) return;
    tblis_sumv_ref(false, n, A, inc_A, sum);
}

template <typename T>
T tblis_sumv(dim_t n, const T* A, inc_t inc_A)
{
    T sum;
    tblis_sumv(n, A, inc_A, sum);
    return sum;
}

template <typename T>
void tblis_sumv(const Matrix<T>& A, T& sum)
{
    Matrix<T> Av;

    ViewNoTranspose(const_cast<Matrix<T>&>(A), Av);

    if (A.length() == 0 || A.width() == 0) return;

    tblis_sumv_int(A, sum);
}

template <typename T>
T tblis_sumv(const Matrix<T>& A)
{
    T sum;
    tblis_sumv(A, sum);
    return sum;
}

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void tblis_sumv(dim_t n, const T* A, inc_t inc_A, T& sum); \
template    T tblis_sumv(dim_t n, const T* A, inc_t inc_A); \
template void tblis_sumv(const Matrix<T>& A, T& sum); \
template    T tblis_sumv(const Matrix<T>& A);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

}
}

#endif
