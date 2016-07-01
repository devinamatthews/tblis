#include "tblis.hpp"

namespace tblis
{

template <typename T>
void tblis_normfm(const_matrix_view<T> A, T& restrict norm)
{
    idx_type m = A.length(0);
    idx_type n = A.length(1);
    stride_type rs = A.stride(0);
    stride_type cs = A.stride(1);
    const T* restrict p = A.data();

    norm = 0;

    for (idx_type i = 0;i < m;i++)
    {
        for (idx_type j = 0;j < n;j++)
        {
            T val = *(p+rs*i+cs*j);
            norm += norm2(val);
        }
    }

    norm = sqrt(real(norm));
}

template <typename T>
void tblis_normfm(const_scatter_matrix_view<T> A, T& restrict norm)
{
    idx_type m = A.length(0);
    idx_type n = A.length(1);
    stride_type rs = A.stride(0);
    stride_type cs = A.stride(1);
    const stride_type* restrict rscat = A.scatter(0);
    const stride_type* restrict cscat = A.scatter(1);
    const T* restrict p = A.data();

    norm = 0;

    if (rs == 0 && cs == 0)
    {
        for (idx_type i = 0;i < m;i++)
        {
            for (idx_type j = 0;j < n;j++)
            {
                T val = *(p+rscat[i]+cscat[j]);
                norm += norm2(val);
            }
        }
    }
    else if (rs == 0)
    {
        for (idx_type i = 0;i < m;i++)
        {
            for (idx_type j = 0;j < n;j++)
            {
                T val = *(p+rscat[i]+cs*j);
                norm += norm2(val);
            }
        }
    }
    else if (cs == 0)
    {
        for (idx_type i = 0;i < m;i++)
        {
            for (idx_type j = 0;j < n;j++)
            {
                T val = *(p+rs*i+cscat[j]);
                norm += norm2(val);
            }
        }
    }
    else
    {
        for (idx_type i = 0;i < m;i++)
        {
            for (idx_type j = 0;j < n;j++)
            {
                T val = *(p+rs*i+cs*j);
                norm += norm2(val);
            }
        }
    }

    norm = sqrt(real(norm));
}

template <typename T>
T tblis_normfm(const_matrix_view<T> A)
{
    T norm;
    tblis_normfm(A, norm);
    return norm;
}

template <typename T>
T tblis_normfm(const_scatter_matrix_view<T> A)
{
    T norm;
    tblis_normfm(A, norm);
    return norm;
}

#define INSTANTIATE_FOR_TYPE(T) \
template void tblis_normfm(        const_matrix_view<T> A, T& norm); \
template void tblis_normfm(const_scatter_matrix_view<T> A, T& norm); \
template    T tblis_normfm(        const_matrix_view<T> A); \
template    T tblis_normfm(const_scatter_matrix_view<T> A);
#include "tblis_instantiate_for_types.hpp"

}
