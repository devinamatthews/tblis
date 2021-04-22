#ifndef _TBLIS_KERNELS_1F_DOTF_HPP_
#define _TBLIS_KERNELS_1F_DOTF_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "util/macros.h"

#include "../1v/dot.hpp"

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

    if (m == NF)
    {
        if (is_complex<T>::value && conj_B)
        {
            if (cs_A == 1 && inc_B == 1)
            {
                #pragma omp simd
                for (len_type j = 0;j < n;j++)
                    for (len_type i = 0;i < NF;i++)
                        AB[i] += A[i*rs_A + j]*conj(B[j]);
            }
            else
            {
                for (len_type j = 0;j < n;j++)
                    for (len_type i = 0;i < NF;i++)
                        AB[i] += A[i*rs_A + j*cs_A]*conj(B[j*inc_B]);
            }
        }
        else
        {
            if (cs_A == 1 && inc_B == 1)
            {
                #pragma omp simd
                for (len_type j = 0;j < n;j++)
                    for (len_type i = 0;i < NF;i++)
                        AB[i] += A[i*rs_A + j]*B[j];
            }
            else
            {
                for (len_type j = 0;j < n;j++)
                    for (len_type i = 0;i < NF;i++)
                        AB[i] += A[i*rs_A + j*cs_A]*B[j*inc_B];
            }
        }
    }
    else
    {
        if (cs_A == 1 && inc_B == 1)
        {
            for (len_type i = 0;i < m;i++)
                #pragma omp simd
                for (len_type j = 0;j < n;j++)
                    AB[i] += A[i*rs_A + j]*conj(conj_B, B[j]);
        }
        else
        {
            for (len_type i = 0;i < m;i++)
                for (len_type j = 0;j < n;j++)
                    AB[i] += A[i*rs_A + j*cs_A]*conj(conj_B, B[j*inc_B]);
        }
    }

    if (beta == T(0))
    {
        for (len_type i = 0;i < m;i++)
            C[i*inc_C] = alpha*conj(conj_A, AB[i]);
    }
    else
    {
        for (len_type i = 0;i < m;i++)
            C[i*inc_C] = alpha*conj(conj_A, AB[i]) +
                         beta*conj(conj_C, C[i*inc_C]);
    }
}

}

#endif
