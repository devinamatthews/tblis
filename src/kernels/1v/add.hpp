#ifndef _TBLIS_KERNELS_1V_ADD_HPP_
#define _TBLIS_KERNELS_1V_ADD_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "util/macros.h"

namespace tblis
{

template <typename T>
using add_ukr_t =
    void (*)(len_type n,
             T alpha, bool conj_A, const T* A, stride_type inc_A,
             T  beta, bool conj_B,       T* B, stride_type inc_B);

template <typename Config, typename T>
void add_ukr_def(len_type n,
                 T alpha, bool conj_A, const T* TBLIS_RESTRICT A, stride_type inc_A,
                 T  beta, bool conj_B,       T* TBLIS_RESTRICT B, stride_type inc_B)
{
    if (beta == T(0))
    {
        if (is_complex<T>::value && conj_A)
        {
            if (inc_A == 1 && inc_B == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    B[i] = alpha*conj(A[i]);
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    B[i*inc_B] = alpha*conj(A[i*inc_A]);
            }
        }
        else
        {
            if (inc_A == 1 && inc_B == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    B[i] = alpha*A[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    B[i*inc_B] = alpha*A[i*inc_A];
            }
        }
    }
    else
    {
        if (is_complex<T>::value && conj_A && conj_B)
        {
            if (inc_A == 1 && inc_B == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    B[i] = alpha*conj(A[i]) + beta*conj(B[i]);
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    B[i*inc_B] = alpha*conj(A[i*inc_A]) + beta*conj(B[i*inc_B]);
            }
        }
        else if (is_complex<T>::value && conj_B)
        {
            if (inc_A == 1 && inc_B == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    B[i] = alpha*A[i] + beta*conj(B[i]);
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    B[i*inc_B] = alpha*A[i*inc_A] + beta*conj(B[i*inc_B]);
            }
        }
        else if (is_complex<T>::value && conj_A)
        {
            if (inc_A == 1 && inc_B == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    B[i] = alpha*conj(A[i]) + beta*B[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    B[i*inc_B] = alpha*conj(A[i*inc_A]) + beta*B[i*inc_B];
            }
        }
        else
        {
            if (inc_A == 1 && inc_B == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    B[i] = alpha*A[i] + beta*B[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    B[i*inc_B] = alpha*A[i*inc_A] + beta*B[i*inc_B];
            }
        }
    }
}

}

#endif
