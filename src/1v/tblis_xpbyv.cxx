#ifndef _TBLIS_XPBYV_HPP_
#define _TBLIS_XPBYV_HPP_

#include "tblis.hpp"

namespace tblis
{

template <typename T>
static void tblis_xpbyv_ref(bool conj_A, idx_type n,
                                     const T* restrict A, stride_type inc_A,
                            T  beta,       T* restrict B, stride_type inc_B)
{
    if (n == 0) return;

    if (inc_A == 1 && inc_B == 1)
    {
        if (conj_A)
        {
            for (idx_type i = 0;i < n;i++)
            {
                B[i] = conj(A[i]) + beta*B[i];
            }
        }
        else
        {
            for (idx_type i = 0;i < n;i++)
            {
                B[i] = A[i] + beta*B[i];
            }
        }
    }
    else
    {
        if (conj_A)
        {
            for (idx_type i = 0;i < n;i++)
            {
                (*B) = conj(*A) + beta*(*B);
                A += inc_A;
                B += inc_B;
            }
        }
        else
        {
            for (idx_type i = 0;i < n;i++)
            {
                (*B) = (*A) + beta*(*B);
                A += inc_A;
                B += inc_B;
            }
        }
    }
}

template <typename T>
void tblis_xpbyv(const_row_view<T> A, T beta, row_view<T> B)
{
    if (beta == T(0))
    {
        tblis_copyv(A, B);
    }
    else if (beta == T(1))
    {
        tblis_addv(A, B);
    }
    else
    {
        assert(A.length() == B.length());
        tblis_xpbyv_ref(false, A.length(), A.data(), A.stride(),
                                     beta, B.data(), B.stride());
    }
}

template <typename T>
void tblis_xpbyv(bool conj_A, idx_type n,
                         const T* A, stride_type inc_A,
                 T beta,       T* B, stride_type inc_B)
{
    if (beta == T(0))
    {
        tblis_copyv(conj_A, n, A, inc_A, B, inc_B);
    }
    else if (beta == T(1))
    {
        tblis_addv(conj_A, n, A, inc_A, B, inc_B);
    }
    else
    {
        tblis_xpbyv_ref(conj_A, n, A, inc_A, beta, B, inc_B);
    }
}

#define INSTANTIATE_FOR_TYPE(T) \
template void tblis_xpbyv(bool conj_A, idx_type n, const T* A, stride_type inc_A, T beta, T* B, stride_type inc_B); \
template void tblis_xpbyv(const_row_view<T> A, T beta, row_view<T> B);
#include "tblis_instantiate_for_types.hpp"

}

#endif
