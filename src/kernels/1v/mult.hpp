#ifndef _TBLIS_KERNELS_1V_MULT_HPP_
#define _TBLIS_KERNELS_1V_MULT_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "util/macros.h"

namespace tblis
{

template <typename T>
using mult_ukr_t =
    void (*)(len_type n,
             T alpha, bool conj_A, const T* A, stride_type inc_A,
                      bool conj_B, const T* B, stride_type inc_B,
             T  beta, bool conj_C,       T* C, stride_type inc_C);

template <typename Config, typename T>
void mult_ukr_def(len_type n,
                  T alpha, bool conj_A, const T* TBLIS_RESTRICT A, stride_type inc_A,
                           bool conj_B, const T* TBLIS_RESTRICT B, stride_type inc_B,
                  T  beta, bool conj_C,       T* TBLIS_RESTRICT C, stride_type inc_C)
{
    if (is_complex<T>::value && conj_B && !conj_A)
    {
        std::swap(conj_A, conj_B);
        std::swap(A, B);
        std::swap(inc_A, inc_B);
    }

    if (beta == T(0))
    {
        if (is_complex<T>::value && conj_A && conj_B)
        {
            if (inc_A == 1 && inc_B == 1 && inc_C == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    C[i] = alpha*conj(A[i])*conj(B[i]);
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    C[i*inc_C] = alpha*conj(A[i*inc_A])*conj(B[i*inc_B]);
            }
        }
        else if (is_complex<T>::value && conj_A)
        {
            if (inc_A == 1 && inc_B == 1 && inc_C == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    C[i] = alpha*conj(A[i])*B[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    C[i*inc_C] = alpha*conj(A[i*inc_A])*B[i*inc_B];
            }
        }
        else
        {
            if (inc_A == 1 && inc_B == 1 && inc_C == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    C[i] = alpha*A[i]*B[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    C[i*inc_C] = alpha*A[i*inc_A]*B[i*inc_B];
            }
        }
    }
    else
    {
        if (is_complex<T>::value && conj_A && conj_B && conj_C)
        {
            if (inc_A == 1 && inc_B == 1 && inc_C == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    C[i] = alpha*conj(A[i])*conj(B[i]) + beta*conj(C[i]);
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    C[i*inc_C] = alpha*conj(A[i*inc_A])*conj(B[i*inc_B]) +
                                  beta*conj(C[i*inc_C]);
            }
        }
        else if (is_complex<T>::value && conj_A && conj_C)
        {
            if (inc_A == 1 && inc_B == 1 && inc_C == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    C[i] = alpha*conj(A[i])*B[i] + beta*conj(C[i]);
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    C[i*inc_C] = alpha*conj(A[i*inc_A])*B[i*inc_B] +
                                  beta*conj(C[i*inc_C]);
            }
        }
        else if (is_complex<T>::value && conj_C)
        {
            if (inc_A == 1 && inc_B == 1 && inc_C == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    C[i] = alpha*A[i]*B[i] + beta*conj(C[i]);
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    C[i*inc_C] = alpha*A[i*inc_A]*B[i*inc_B] +
                                  beta*conj(C[i*inc_C]);
            }
        }
        else if (is_complex<T>::value && conj_A && conj_B)
        {
            if (inc_A == 1 && inc_B == 1 && inc_C == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    C[i] = alpha*conj(A[i])*conj(B[i]) + beta*C[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    C[i*inc_C] = alpha*conj(A[i*inc_A])*conj(B[i*inc_B]) +
                                  beta*C[i*inc_C];
            }
        }
        else if (is_complex<T>::value && conj_A)
        {
            if (inc_A == 1 && inc_B == 1 && inc_C == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    C[i] = alpha*conj(A[i])*B[i] + beta*C[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    C[i*inc_C] = alpha*conj(A[i*inc_A])*B[i*inc_B] +
                                  beta*C[i*inc_C];
            }
        }
        else
        {
            if (inc_A == 1 && inc_B == 1 && inc_C == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    C[i] = alpha*A[i]*B[i] + beta*C[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    C[i*inc_C] = alpha*A[i*inc_A]*B[i*inc_B] + beta*C[i*inc_C];
            }
        }
    }
}

}

#endif
