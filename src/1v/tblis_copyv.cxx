#include "tblis.hpp"

namespace tblis
{

template <typename T>
static void tblis_copyv_ref(bool conj_A, idx_type n,
                            const T* restrict A, stride_type inc_A,
                                  T* restrict B, stride_type inc_B)
{
    if (n == 0) return;

    if (inc_A == 1 && inc_B == 1)
    {
        if (conj_A)
        {
            for (idx_type i = 0;i < n;i++)
            {
                B[i] = conj(A[i]);
            }
        }
        else
        {
            for (idx_type i = 0;i < n;i++)
            {
                B[i] = A[i];
            }
        }
    }
    else
    {
        if (conj_A)
        {
            for (idx_type i = 0;i < n;i++)
            {
                (*B) = conj(*A);
                A += inc_A;
                B += inc_B;
            }
        }
        else
        {
            for (idx_type i = 0;i < n;i++)
            {
                (*B) = (*A);
                A += inc_A;
                B += inc_B;
            }
        }
    }
}

template <typename T>
void tblis_copyv(const_row_view<T> A, row_view<T> B)
{
    assert(A.length() == B.length());
    tblis_copyv_ref(false, A.length(), A.data(), A.stride(),
                                      B.data(), B.stride());
}

template <typename T>
void tblis_copyv(bool conj_A, idx_type n,
                 const T* A, stride_type inc_A,
                       T* B, stride_type inc_B)
{
    tblis_copyv_ref(conj_A, n, A, inc_A, B, inc_B);
}

#define INSTANTIATE_FOR_TYPE(T) \
template void tblis_copyv(bool conj_A, idx_type n, const T* A, stride_type inc_A, T* B, stride_type inc_B); \
template void tblis_copyv(const_row_view<T> A, row_view<T> B);
#include "tblis_instantiate_for_types.hpp"

}
