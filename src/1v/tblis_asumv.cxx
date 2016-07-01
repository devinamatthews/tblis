#include "tblis.hpp"

namespace tblis
{

template <typename T>
static void tblis_asumv_ref(idx_type n, const T* restrict A, stride_type inc_A, T& restrict sum)
{
    sum = T();

    if (n == 0) return;

    if (inc_A == 1)
    {
        for (idx_type i = 0;i < n;i++)
        {
            sum += std::abs(A[i]);
        }
    }
    else
    {
        for (idx_type i = 0;i < n;i++)
        {
            sum += std::abs(*A);
            A += inc_A;
        }
    }
}

template <typename T>
void tblis_asumv(const_row_view<T> A, T& sum)
{
    tblis_asumv_ref(A.length(), A.data(), A.stride(), sum);
}

template <typename T>
T tblis_asumv(const_row_view<T> A)
{
    T sum;
    tblis_asumv(A, sum);
    return sum;
}

template <typename T>
void tblis_asumv(idx_type n, const T* A, stride_type inc_A, T& sum)
{
    tblis_asumv_ref(n, A, inc_A, sum);
}

template <typename T>
T tblis_asumv(idx_type n, const T* A, stride_type inc_A)
{
    T sum;
    tblis_asumv(n, A, inc_A, sum);
    return sum;
}

#define INSTANTIATE_FOR_TYPE(T) \
template void tblis_asumv(idx_type n, const T* A, stride_type inc_A, T& sum); \
template    T tblis_asumv(idx_type n, const T* A, stride_type inc_A); \
template void tblis_asumv(const_row_view<T> A, T& sum); \
template    T tblis_asumv(const_row_view<T> A);
#include "tblis_instantiate_for_types.hpp"

}
