#ifndef _TBLIS_KERNELS_1V_SHIFT_HPP_
#define _TBLIS_KERNELS_1V_SHIFT_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "util/macros.h"

namespace tblis
{

template <typename T>
using shift_ukr_t =
    void (*)(len_type n,
             T alpha, T beta, bool conj_A, T* A, stride_type inc_A);

template <typename Config, typename T>
void shift_ukr_def(len_type n,
                   T alpha, T beta, bool conj_A, T* A, stride_type inc_A)
{
    if (beta == T(0))
    {
        if (inc_A == 1)
        {
            for (len_type i = 0;i < n;i++) A[i] = alpha;
        }
        else
        {
            for (len_type i = 0;i < n;i++) A[i*inc_A] = alpha;
        }
    }
    else
    {
        if (is_complex<T>::value && conj_A)
        {
            if (inc_A == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    A[i] = alpha + beta*conj(A[i]);
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    A[i*inc_A] = alpha + beta*conj(A[i*inc_A]);
            }
        }
        else
        {
            if (inc_A == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    A[i] = alpha + beta*A[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    A[i*inc_A] = alpha + beta*A[i*inc_A];
            }
        }
    }
}

}

#endif
