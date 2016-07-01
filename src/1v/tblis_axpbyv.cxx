#include "tblis.hpp"

namespace tblis
{

template <typename T>
static void tblis_axpbyv_ref(bool conj_A, idx_type n,
                             T alpha, const T* restrict A, stride_type inc_A,
                             T  beta,       T* restrict B, stride_type inc_B)
{
    if (n == 0) return;

    if (inc_A == 1 && inc_B == 1)
    {
        if (conj_A)
        {
            for (idx_type i = 0;i < n;i++)
            {
                B[i] = alpha*conj(A[i]) + beta*B[i];
            }
        }
        else
        {
            for (idx_type i = 0;i < n;i++)
            {
                B[i] = alpha*A[i] + beta*B[i];
            }
        }
    }
    else
    {
        if (conj_A)
        {
            for (idx_type i = 0;i < n;i++)
            {
                (*B) = alpha*conj(*A) + beta*(*B);
                A += inc_A;
                B += inc_B;
            }
        }
        else
        {
            for (idx_type i = 0;i < n;i++)
            {
                (*B) = alpha*(*A) + beta*(*B);
                A += inc_A;
                B += inc_B;
            }
        }
    }
}

template <typename T>
void tblis_axpbyv(T alpha, const_row_view<T> A, T beta, row_view<T> B)
{
    if (alpha == T(0))
    {
        tblis_scalv(beta, B);
    }
    else if (alpha == T(1))
    {
        tblis_xpbyv(A, beta, B);
    }
    else if (beta == T(0))
    {
        tblis_scal2v(alpha, A, B);
    }
    else if (beta == T(1))
    {
        tblis_axpyv(alpha, A, B);
    }
    else
    {
        assert(A.length() == B.length());
        tblis_axpbyv_ref(false, A.length(), alpha, A.data(), A.stride(),
                                             beta, B.data(), B.stride());
    }
}
template <typename T>
void tblis_axpbyv(bool conj_A, idx_type n,
                  T alpha, const T* A, idx_type inc_A,
                  T  beta,       T* B, idx_type inc_B)
{
    if (alpha == T(0))
    {
        tblis_scalv(n, beta, B, inc_B);
    }
    else if (alpha == T(1))
    {
        tblis_xpbyv(conj_A, n, A, inc_A, beta, B, inc_B);
    }
    else if (beta == T(0))
    {
        tblis_scal2v(conj_A, n, alpha, A, inc_A, B, inc_B);
    }
    else if (beta == T(1))
    {
        tblis_axpyv(conj_A, n, alpha, A, inc_A, B, inc_B);
    }
    else
    {
        tblis_axpbyv_ref(conj_A, n, alpha, A, inc_A, beta, B, inc_B);
    }
}

#define INSTANTIATE_FOR_TYPE(T) \
template void tblis_axpbyv(bool conj_A, idx_type n, T alpha, const T* A, idx_type inc_A, T beta, T* B, idx_type inc_B); \
template void tblis_axpbyv(T alpha, const_row_view<T> A, T beta, row_view<T> B);
#include "tblis_instantiate_for_types.hpp"

}
