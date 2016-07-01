#include "tblis.hpp"

namespace tblis
{

template <typename T>
static void tblis_dotv_ref(bool conj_A, bool conj_B, idx_type n,
                           const T* restrict A, stride_type inc_A,
                           const T* restrict B, stride_type inc_B, T& restrict dot)
{
    dot = T();

    if (n == 0) return;

    conj_B = conj_A ^ conj_B;

    if (inc_A == 1)
    {
        if (conj_B)
        {
            for (idx_type i = 0;i < n;i++)
            {
                dot += A[i]*conj(B[i]);
            }
        }
        else
        {
            for (idx_type i = 0;i < n;i++)
            {
                dot += A[i]*B[i];
            }
        }
    }
    else
    {
        if (conj_B)
        {
            for (idx_type i = 0;i < n;i++)
            {
                dot += (*A)*conj(*B);
                A += inc_A;
                B += inc_B;
            }
        }
        else
        {
            for (idx_type i = 0;i < n;i++)
            {
                dot += (*A)*(*B);
                A += inc_A;
                B += inc_B;
            }
        }
    }

    if (conj_A) dot = conj(dot);
}

template <typename T>
void tblis_dotv(const_row_view<T> A, const_row_view<T> B, T& dot)
{
    assert(A.length() == B.length());
    tblis_dotv_ref(false, false,
                   A.length(), A.data(), A.stride(),
                               B.data(), B.stride(), dot);
}

template <typename T>
T tblis_dotv(const_row_view<T> A, const_row_view<T> B)
{
    T dot;
    tblis_dotv(A, B, dot);
    return dot;
}

template <typename T>
void tblis_dotv(bool conj_A, bool conj_B, idx_type n,
                const T* A, stride_type inc_A,
                const T* B, stride_type inc_B, T& dot)
{
    tblis_dotv_ref(conj_A, conj_B, n, A, inc_A, B, inc_B, dot);
}

template <typename T>
T tblis_dotv(bool conj_A, bool conj_B, idx_type n,
             const T* A, stride_type inc_A,
             const T* B, stride_type inc_B)
{
    T dot;
    tblis_dotv(conj_A, conj_B, n, A, inc_A, B, inc_B, dot);
    return dot;
}

#define INSTANTIATE_FOR_TYPE(T) \
template void tblis_dotv(bool conj_A, bool conj_B, idx_type n, const T* A, stride_type inc_A, const T* B, stride_type inc_B, T& dot); \
template    T tblis_dotv(bool conj_A, bool conj_B, idx_type n, const T* A, stride_type inc_A, const T* B, stride_type inc_B); \
template void tblis_dotv(const_row_view<T> A, const_row_view<T> B, T& dot); \
template    T tblis_dotv(const_row_view<T> A, const_row_view<T> B);
#include "tblis_instantiate_for_types.hpp"

}
