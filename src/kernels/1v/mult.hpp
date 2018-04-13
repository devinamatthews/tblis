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
    TBLIS_SPECIAL_CASE(alpha == T(1),
    TBLIS_SPECIAL_CASE(beta == T(1),
    TBLIS_SPECIAL_CASE(is_complex<T>::value && conj_A,
    TBLIS_SPECIAL_CASE(is_complex<T>::value && conj_B,
    TBLIS_SPECIAL_CASE(is_complex<T>::value && conj_C,
    TBLIS_SPECIAL_CASE(inc_A == 1 && inc_B == 1 && inc_C == 1,
    {
        for (int i = 0;i < n;i++)
            C[i*inc_C] = alpha*(conj_A ? conj(A[i*inc_A]) : A[i*inc_A])*
                               (conj_B ? conj(B[i*inc_B]) : B[i*inc_B])+
                          beta*(conj_C ? conj(C[i*inc_C]) : C[i*inc_C]);
    }
    ))))))
}

}

#endif
