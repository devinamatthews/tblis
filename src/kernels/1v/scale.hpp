#ifndef _TBLIS_KERNELS_1V_SCALE_HPP_
#define _TBLIS_KERNELS_1V_SCALE_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "util/macros.h"

namespace tblis
{

using scale_ukr_t =
    void (*)(len_type n,
             const void* alpha, bool conj_A, void* A, stride_type inc_A);

template <typename Config, typename T>
void scale_ukr_def(len_type n,
                   const void* alpha_, bool conj_A, void* A_, stride_type inc_A)
{
    T alpha = *static_cast<const T*>(alpha_);

    T* TBLIS_RESTRICT A = static_cast<T*>(A_);

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
        if (is_complex<T>::value && conj_A)
        {
            if (inc_A == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    A[i] = alpha*conj(A[i]);
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    A[i*inc_A] = alpha*conj(A[i*inc_A]);
            }
        }
        else
        {
            if (inc_A == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    A[i] = alpha*A[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    A[i*inc_A] = alpha*A[i*inc_A];
            }
        }
    }
}

}

#endif
