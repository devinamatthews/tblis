#ifndef _TBLIS_KERNELS_1V_SET_HPP_
#define _TBLIS_KERNELS_1V_SET_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "util/macros.h"

namespace tblis
{

template <typename T>
using set_ukr_t =
    void (*)(const communicator& comm, len_type n,
             T alpha, T* A, stride_type inc_A);

template <typename T>
void set_ukr_def(const communicator& comm, len_type n,
                 T alpha, T* A, stride_type inc_A)
{
    len_type first, last;
    std::tie(first, last, std::ignore) = comm.distribute_over_threads(n);

    A += first*inc_A;
    n = last-first;

    TBLIS_SPECIAL_CASE(inc_A == 1,
    {
        for (int i = 0;i < n;i++) A[i*inc_A] = alpha;
    })
}

}

#endif
