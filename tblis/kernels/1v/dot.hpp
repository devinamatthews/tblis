#ifndef TBLIS_KERNELS_1V_DOT_HPP
#define TBLIS_KERNELS_1V_DOT_HPP 1

#include <tblis/internal/types.hpp>

namespace tblis
{

template <typename Config, typename T>
void dot_ukr_def(len_type n,
                 bool conj_A, const void* A_, stride_type inc_A,
                 bool conj_B, const void* B_, stride_type inc_B, void* value_)
{
    const T* TBLIS_RESTRICT A = static_cast<const T*>(A_);
    const T* TBLIS_RESTRICT B = static_cast<const T*>(B_);

    T value = *static_cast<T*>(value_);

    if (conj_A)
    {
        value = conj(value);
        conj_B = !conj_B;
    }

    if (is_complex<T>::value && conj_B)
    {
        if (inc_A == 1 && inc_B == 1)
        {
            #pragma omp simd
            for (len_type i = 0;i < n;i++)
                value += A[i]*conj(B[i]);
        }
        else
        {
            for (len_type i = 0;i < n;i++)
                value += A[i*inc_A]*conj(B[i*inc_B]);
        }
    }
    else
    {
        if (inc_A == 1 && inc_B == 1)
        {
            #pragma omp simd
            for (len_type i = 0;i < n;i++)
                value += A[i]*B[i];
        }
        else
        {
            for (len_type i = 0;i < n;i++)
                value += A[i*inc_A]*B[i*inc_B];
        }
    }

    if (conj_A)
    {
        value = conj(value);
    }

    *static_cast<T*>(value_) = value;
}

}

#endif //TBLIS_KERNELS_1V_DOT_HPP
