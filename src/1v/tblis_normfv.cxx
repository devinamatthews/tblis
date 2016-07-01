#include "tblis.hpp"

namespace tblis
{

template <typename T>
static void tblis_normfv_ref(idx_type n, const T* restrict A, stride_type inc_A, T& restrict norm)
{
    norm = T();

    if (n == 0) return;

    if (inc_A == 1)
    {
        for (idx_type i = 0;i < n;i++)
        {
            norm += norm2(A[i]);
        }
    }
    else
    {
        for (idx_type i = 0;i < n;i++)
        {
            norm += norm2(*A);
            A += inc_A;
        }
    }

    norm = sqrt(real(norm));
}

template <typename T>
void tblis_normfv(const_row_view<T> A, T& norm)
{
    tblis_normfv_ref(A.length(), A.data(), A.stride(), norm);
}

template <typename T>
T tblis_normfv(const_row_view<T> A)
{
    T norm;
    tblis_normfv(A, norm);
    return norm;
}

template <typename T>
void tblis_normfv(idx_type n, const T* A, stride_type inc_A, T& norm)
{
    tblis_normfv_ref(n, A, inc_A, norm);
}

template <typename T>
T tblis_normfv(idx_type n, const T* A, stride_type inc_A)
{
    T norm;
    tblis_normfv(n, A, inc_A, norm);
    return norm;
}

#define INSTANTIATE_FOR_TYPE(T) \
template void tblis_normfv(idx_type n, const T* A, stride_type inc_A, T& norm); \
template    T tblis_normfv(idx_type n, const T* A, stride_type inc_A); \
template void tblis_normfv(const_row_view<T> A, T& norm); \
template    T tblis_normfv(const_row_view<T> A);
#include "tblis_instantiate_for_types.hpp"

}
