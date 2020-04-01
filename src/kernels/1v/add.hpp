#ifndef _TBLIS_KERNELS_1V_ADD_HPP_
#define _TBLIS_KERNELS_1V_ADD_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "util/macros.h"

#pragma clang diagnostic ignored "-Wpass-failed"

namespace tblis
{

using add_ukr_t =
    void (*)(len_type n,
             const void* alpha, bool conj_A, const void* A, stride_type inc_A,
             const void*  beta, bool conj_B,       void* B, stride_type inc_B);

template <typename Config, typename T>
void add_ukr_def(len_type n,
                 const void* alpha_, bool conj_A, const void* A_, stride_type inc_A,
                 const void*  beta_, bool conj_B,       void* B_, stride_type inc_B)
{
    T alpha = *static_cast<const T*>(alpha_);
    T beta  = *static_cast<const T*>(beta_ );

    const T* TBLIS_RESTRICT A = static_cast<const T*>(A_);
          T* TBLIS_RESTRICT B = static_cast<      T*>(B_);

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
