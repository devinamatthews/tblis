#include "tblis.hpp"

namespace tblis
{

template <typename T>
static void tblis_zerov_ref(idx_type n, T* restrict A, stride_type inc_A)
{
    if (n == 0) return;

    if (inc_A == 1)
    {
        for (idx_type i = 0;i < n;i++)
        {
            A[i] = T();
        }
    }
    else
    {
        for (idx_type i = 0;i < n;i++)
        {
            (*A) = T();
            A += inc_A;
        }
    }
}

template <typename T>
void tblis_zerov(row_view<T> A)
{
    tblis_zerov_ref(A.length(), A.data(), A.stride());
}

template <typename T>
void tblis_zerov(idx_type n, T* A, stride_type inc_A)
{
    tblis_zerov_ref(n, A, inc_A);
}

#define INSTANTIATE_FOR_TYPE(T) \
template void tblis_zerov(idx_type n, T* A, stride_type inc_A); \
template void tblis_zerov(row_view<T> A);
#include "tblis_instantiate_for_types.hpp"

}
