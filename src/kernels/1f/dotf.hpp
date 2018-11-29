#ifndef _TBLIS_KERNELS_1F_DOTF_HPP_
#define _TBLIS_KERNELS_1F_DOTF_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "util/macros.h"

namespace tblis
{

template <typename T>
using dotf_ukr_t =
    void (*)(len_type m, len_type n,
             T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
                      bool conj_B, const T* B, stride_type inc_B,
             T  beta, bool conj_C,       T* C, stride_type inc_C);

template <typename Config, typename T>
void dotf_ukr_def(len_type m, len_type n,
                  T alpha, bool conj_A, const T* TBLIS_RESTRICT A, stride_type rs_A, stride_type cs_A,
                           bool conj_B, const T* TBLIS_RESTRICT B, stride_type inc_B,
                  T  beta, bool conj_C,       T* TBLIS_RESTRICT C, stride_type inc_C)
{
    constexpr len_type NF = Config::template dotf_nf<T>::def;

    T AB[NF] = {};

    if (conj_A) conj_B = !conj_B;

    TBLIS_SPECIAL_CASE(is_complex<T>::value && conj_B,
    TBLIS_SPECIAL_CASE(m == NF,
    {
        if (cs_A == 1 && inc_B == 1)
        {
            #pragma omp simd
            for (len_type j = 0;j < n;j++)
                for (len_type i = 0;i < m;i++)
                    AB[i] += A[i*rs_A + j]*(conj_B ? conj(B[j]) : B[j]);
        }
        else
        {
            for (len_type j = 0;j < n;j++)
                for (len_type i = 0;i < m;i++)
                    AB[i] += A[i*rs_A + j*cs_A]*(conj_B ? conj(B[j*inc_B]) : B[j*inc_B]);
        }
    }
    ))

    if (beta == T(0))
    {
        for (len_type i = 0;i < m;i++)
            C[i*inc_C] = alpha*(conj_A ? conj(AB[i]) : AB[i]);
    }
    else
    {
        for (len_type i = 0;i < m;i++)
            C[i*inc_C] = alpha*(conj_A ? conj(AB[i]) : AB[i]) +
                         beta*(conj_C ? conj(C[i*inc_C]) : C[i*inc_C]);
    }
}

}

#endif
