#ifndef _TBLIS_KERNELS_1V_SCALE_HPP_
#define _TBLIS_KERNELS_1V_SCALE_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "util/macros.h"

namespace tblis
{

template <typename T>
using scale_ukr_t =
    void (*)(len_type n,
             T alpha, bool conj_A, T* A, stride_type inc_A);

template <typename Config, typename T>
void scale_ukr_def(len_type n,
                   T alpha, bool conj_A, T* A, stride_type inc_A)
{
    if (alpha == T(0))
    {
        if (inc_A == 1)
        {
            for (len_type i = 0;i < n;i++) A[i] = T(0);
        }
        else
        {
            for (len_type i = 0;i < n;i++) A[i*inc_A] = T(0);
        }
    }
    else
    {
        TBLIS_SPECIAL_CASE(is_complex<T>::value && conj_A,
        {
            if (inc_A == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    A[i] = alpha*(conj_A ? conj(A[i]) : A[i]);
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    A[i*inc_A] = alpha*(conj_A ? conj(A[i*inc_A]) : A[i*inc_A]);
            }
        }
        )
    }
}

}

#endif
