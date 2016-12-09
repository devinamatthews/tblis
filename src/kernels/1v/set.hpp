#ifndef _TBLIS_KERNELS_1V_SET_HPP_
#define _TBLIS_KERNELS_1V_SET_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "util/macros.h"

namespace tblis
{

template <typename T>
using set_ukr_t =
    void (*)(len_type n,
             T alpha, T* A, stride_type inc_A);

template <typename Config, typename T>
void set_ukr_def(len_type n,
                 T alpha, T* A, stride_type inc_A)
{
    TBLIS_SPECIAL_CASE(inc_A == 1,
    {
        for (int i = 0;i < n;i++) A[i*inc_A] = alpha;
    })
}

}

#endif
