#ifndef _TBLIS_KERNELS_1V_ADD_HPP_
#define _TBLIS_KERNELS_1V_ADD_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "util/macros.h"

namespace tblis
{

template <typename T>
using add_ukr_t =
    void (*)(const communicator& comm, len_type n,
             T alpha, bool conj_A, const T* A, stride_type inc_A,
             T  beta, bool conj_B,       T* B, stride_type inc_B);

template <typename T>
void add_ukr_def(const communicator& comm, len_type n,
                 T alpha, bool conj_A, const T* TBLIS_RESTRICT A, stride_type inc_A,
                 T  beta, bool conj_B,       T* TBLIS_RESTRICT B, stride_type inc_B)
{
    len_type first, last;
    std::tie(first, last, std::ignore) = comm.distribute_over_threads(n);

    A += first*inc_A;
    B += first*inc_B;
    n = last-first;

    TBLIS_SPECIAL_CASE(alpha == T(1),
    TBLIS_SPECIAL_CASE(beta == T(1),
    TBLIS_SPECIAL_CASE(is_complex<T>::value && conj_A,
    TBLIS_SPECIAL_CASE(is_complex<T>::value && conj_B,
    TBLIS_SPECIAL_CASE(inc_A == 1 && inc_B == 1,
    {
        for (int i = 0;i < n;i++)
            B[i*inc_B] = alpha*(conj_A ? conj(A[i*inc_A]) : A[i*inc_A]) +
                          beta*(conj_B ? conj(B[i*inc_B]) : B[i*inc_B]);
    }
    )))))
}

}

#endif
