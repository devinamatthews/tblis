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
        TBLIS_SPECIAL_CASE(inc_A == 1,
        {
            for (int i = 0;i < n;i++) A[i*inc_A] = alpha;
        }
        )
    }
    else
    {
        TBLIS_SPECIAL_CASE(conj_A,
        TBLIS_SPECIAL_CASE(inc_A == 1,
        {
            for (int i = 0;i < n;i++)
                A[i*inc_A] = alpha + beta*(conj_A ? conj(A[i*inc_A]) : A[i*inc_A]);
        }
        ))
    }
}

}

#endif
