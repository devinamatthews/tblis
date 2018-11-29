#ifndef _TBLIS_KERNELS_1F_ADDF_HPP_
#define _TBLIS_KERNELS_1F_ADDF_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "util/macros.h"

namespace tblis
{

template <typename T>
using addf_ukr_t =
    void (*)(len_type m, len_type n,
             T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
                      bool conj_B, const T* B, stride_type inc_B,
             T  beta, bool conj_C,       T* C, stride_type inc_C);

template <typename Config, typename T>
void addf_ukr_def(len_type m, len_type n,
                  T alpha, bool conj_A, const T* TBLIS_RESTRICT A, stride_type rs_A, stride_type cs_A,
                           bool conj_B, const T* TBLIS_RESTRICT B, stride_type inc_B,
                  T  beta, bool conj_C,       T* TBLIS_RESTRICT C, stride_type inc_C)
{
    constexpr len_type NF = Config::template addf_nf<T>::def;

    T alpha_B[NF];

    for (len_type i = 0;i < n;i++)
        alpha_B[i] = alpha*conj(conj_B, B[i*inc_B]);

    if (n == NF)
    {
        if (beta == T(0))
        {
            if (is_complex<T>::value && conj_A)
            {
                if (rs_A == 1 && inc_C == 1)
                {
                    #pragma omp simd
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i] = T(0);

                        for (len_type j = 0;j < n;j++)
                            C[i] += alpha_B[j]*conj(A[i + j*cs_A]);
                    }
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i*inc_C] = T(0);

                        for (len_type j = 0;j < n;j++)
                            C[i*inc_C] += alpha_B[j]*conj(A[i*rs_A + j*cs_A]);
                    }
                }
            }
            else
            {
                if (rs_A == 1 && inc_C == 1)
                {
                    #pragma omp simd
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i] = T(0);

                        for (len_type j = 0;j < n;j++)
                            C[i] += alpha_B[j]*A[i + j*cs_A];
                    }
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i*inc_C] = T(0);

                        for (len_type j = 0;j < n;j++)
                            C[i*inc_C] += alpha_B[j]*A[i*rs_A + j*cs_A];
                    }
                }
            }
        }
        else
        {
            if (is_complex<T>::value && conj_A && conj_C)
            {
                if (rs_A == 1 && inc_C == 1)
                {
                    #pragma omp simd
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i] = beta*conj(C[i]);

                        for (len_type j = 0;j < n;j++)
                            C[i] += alpha_B[j]*conj(A[i + j*cs_A]);
                    }
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i*inc_C] = beta*conj(C[i*inc_C]);

                        for (len_type j = 0;j < n;j++)
                            C[i*inc_C] += alpha_B[j]*conj(A[i*rs_A + j*cs_A]);
                    }
                }
            }
            else if (is_complex<T>::value && conj_C)
            {
                if (rs_A == 1 && inc_C == 1)
                {
                    #pragma omp simd
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i] = beta*conj(C[i]);

                        for (len_type j = 0;j < n;j++)
                            C[i] += alpha_B[j]*A[i + j*cs_A];
                    }
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i*inc_C] = beta*conj(C[i*inc_C]);

                        for (len_type j = 0;j < n;j++)
                            C[i*inc_C] += alpha_B[j]*A[i*rs_A + j*cs_A];
                    }
                }
            }
            else if (is_complex<T>::value && conj_A)
            {
                if (rs_A == 1 && inc_C == 1)
                {
                    #pragma omp simd
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i] = beta*C[i];

                        for (len_type j = 0;j < n;j++)
                            C[i] += alpha_B[j]*conj(A[i + j*cs_A]);
                    }
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i*inc_C] = beta*C[i*inc_C];

                        for (len_type j = 0;j < n;j++)
                            C[i*inc_C] += alpha_B[j]*conj(A[i*rs_A + j*cs_A]);
                    }
                }
            }
            else
            {
                if (rs_A == 1 && inc_C == 1)
                {
                    #pragma omp simd
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i] = beta*C[i];

                        for (len_type j = 0;j < n;j++)
                            C[i] += alpha_B[j]*A[i + j*cs_A];
                    }
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i*inc_C] = beta*C[i*inc_C];

                        for (len_type j = 0;j < n;j++)
                            C[i*inc_C] += alpha_B[j]*A[i*rs_A + j*cs_A];
                    }
                }
            }
        }
    }
    else
    {
        if (beta == T(0))
        {
            for (len_type i = 0;i < m;i++)
                C[i*inc_C] = T(0);
        }
        else
        {
            for (len_type i = 0;i < m;i++)
                C[i*inc_C] = beta*conj(conj_C, C[i*inc_C]);
        }

        for (len_type j = 0;j < n;j++)
            for (len_type i = 0;i < m;i++)
                C[i*inc_C] += alpha_B[j]*conj(conj_A, A[i*rs_A + j*cs_A]);
    }
}

}

#endif
