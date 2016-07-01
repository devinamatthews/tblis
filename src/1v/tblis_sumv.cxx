#ifndef _TBLIS_SUMV_HPP_
#define _TBLIS_SUMV_HPP_

#include "tblis.hpp"

namespace tblis
{

template <typename T>
static void tblis_sumv_ref(bool conj_A, idx_type n,
                           const T* restrict A, stride_type inc_A, T& restrict sum)
{
    sum = T();

    if (n == 0) return;

    if (inc_A == 1)
    {
        for (idx_type i = 0;i < n;i++)
        {
            sum += A[i];
        }
    }
    else
    {
        for (idx_type i = 0;i < n;i++)
        {
            sum += (*A);
            A += inc_A;
        }
    }

    if (conj_A) sum = conj(sum);
}

template <typename T>
void tblis_sumv(const_row_view<T> A, T& sum)
{
    tblis_sumv_ref(false, A.length(), A.data(), A.stride(), sum);
}

template <typename T>
T tblis_sumv(const_row_view<T> A)
{
    T sum;
    tblis_sumv(A, sum);
    return sum;
}

template <typename T>
void tblis_sumv(idx_type n, const T* A, stride_type inc_A, T& sum)
{
    tblis_sumv_ref(false, n, A, inc_A, sum);
}

template <typename T>
T tblis_sumv(idx_type n, const T* A, stride_type inc_A)
{
    T sum;
    tblis_sumv(n, A, inc_A, sum);
    return sum;
}

#define INSTANTIATE_FOR_TYPE(T) \
template void tblis_sumv(idx_type n, const T* A, stride_type inc_A, T& sum); \
template    T tblis_sumv(idx_type n, const T* A, stride_type inc_A); \
template void tblis_sumv(const_row_view<T> A, T& sum); \
template    T tblis_sumv(const_row_view<T> A);
#include "tblis_instantiate_for_types.hpp"

}

#endif
