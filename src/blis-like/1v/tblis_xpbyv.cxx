#ifndef _TBLIS_XPBYV_HPP_
#define _TBLIS_XPBYV_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T>
static void tblis_xpbyv_ref(bool conj_A, dim_t n,
                                     const T* restrict A, inc_t inc_A,
                            T  beta,       T* restrict B, inc_t inc_B)
{
    if (inc_A == 1 && inc_B == 1)
    {
        if (conj_A)
        {
            for (dim_t i = 0;i < n;i++)
            {
                B[i] = conj(A[i]) + beta*B[i];
            }
        }
        else
        {
            for (dim_t i = 0;i < n;i++)
            {
                B[i] = A[i] + beta*B[i];
            }
        }
    }
    else
    {
        if (conj_A)
        {
            for (dim_t i = 0;i < n;i++)
            {
                (*B) = conj(*A) + beta*(*B);
                A += inc_A;
                B += inc_B;
            }
        }
        else
        {
            for (dim_t i = 0;i < n;i++)
            {
                (*B) = (*A) + beta*(*B);
                A += inc_A;
                B += inc_B;
            }
        }
    }
}

template <typename T>
static void tblis_xpbyv_int(const Matrix<T>& A, T beta, Matrix<T>& B)
{
    if (A.length() > 1)
    {
        tblis_xpbyv_ref(A.is_conjugated(), A.length(), A.data(), A.row_stride(),
                                                 beta, B.data(), B.row_stride());
    }
    else
    {
        tblis_xpbyv_ref(A.is_conjugated(), A.width(), A.data(), A.col_stride(),
                                                beta, B.data(), B.col_stride());
    }
}

template <typename T>
void tblis_xpbyv(bool conj_A, dim_t n,
                         const T* A, inc_t inc_A,
                 T beta,       T* B, inc_t inc_B)
{
    if (beta == 0)
    {
        tblis_copyv(conj_A, n, A, inc_A, B, inc_B);
    }
    else if (beta == 1)
    {
        tblis_addv(conj_A, n, A, inc_A, B, inc_B);
    }

    if (n == 0) return;

    tblis_xpbyv_ref(conj_A, n, A, inc_A, beta, B, inc_B);
}

template <typename T>
void tblis_xpbyv(const Matrix<T>& A, T beta, Matrix<T>& B)
{
    if (beta == 0)
    {
        tblis_copyv(A, B);
    }
    else if (beta == 1)
    {
        tblis_addv(A, B);
    }

    Matrix<T> Av;
    Matrix<T> Bv;

    ViewNoTranspose(const_cast<Matrix<T>&>(A), Av);
    ViewNoTranspose(                       B , Bv);

    ASSERT(A.length() == B.length());
    ASSERT(A.width() == B.width());

    if (A.length() == 0 || A.width() == 0) return;

    tblis_xpbyv_int(A, beta, B);
}

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void tblis_xpbyv(bool conj_A, dim_t n, const T* A, inc_t inc_A, T beta, T* B, inc_t inc_B); \
template void tblis_xpbyv(const Matrix<T>& A, T beta, Matrix<T>& B);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

}
}

#endif
