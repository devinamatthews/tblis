#ifndef _TBLIS_KERNELS_1V_DOT_HPP_
#define _TBLIS_KERNELS_1V_DOT_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "util/macros.h"

namespace tblis
{

template <typename T>
using dot_ukr_t =
    void (*)(len_type n,
             bool conj_A, const T* A, stride_type inc_A,
             bool conj_B, const T* B, stride_type inc_B, T& value);

template <typename Config, typename T>
void dot_ukr_def(len_type n,
                 bool conj_A, const T* TBLIS_RESTRICT A, stride_type inc_A,
                 bool conj_B, const T* TBLIS_RESTRICT B, stride_type inc_B, T& value)
{
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
}

}

#endif
