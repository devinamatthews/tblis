#ifndef _TBLIS_KERNELS_3M_GEMM_HPP_
#define _TBLIS_KERNELS_3M_GEMM_HPP_

#include "util/basic_types.h"
#include <type_traits>

namespace tblis
{

#define EXTERN_GEMM_UKR(T, name) \
extern void name(tblis::stride_type k, \
                 const T* alpha, \
                 const T* a, const T* b, \
                 const T* beta, \
                 T* c, tblis::stride_type rs_c, \
                       tblis::stride_type cs_c);

template <typename T>
using gemm_ukr_t =
void (*)(stride_type k,
        const T* alpha,
        const T* a, const T* b,
        const T* beta,
        T* c, stride_type rs_c, stride_type cs_c);

template <typename Config, typename T>
void gemm_ukr_def(stride_type k,
                  const T* TBLIS_RESTRICT alpha,
                  const T* TBLIS_RESTRICT p_a, const T* TBLIS_RESTRICT p_b,
                  const T* TBLIS_RESTRICT beta,
                  T* TBLIS_RESTRICT p_c, stride_type rs_c, stride_type cs_c)
{
    constexpr len_type MR = Config::template gemm_mr<T>::def;
    constexpr len_type NR = Config::template gemm_nr<T>::def;

    T p_ab[MR*NR] __attribute__((aligned(64))) = {};

    while (k --> 0)
    {
        for (int i = 0;i < MR;i++)
            #pragma omp simd
            for (int j = 0;j < NR;j++)
                p_ab[i*NR + j] += p_a[i] * p_b[j];

        p_a += MR;
        p_b += NR;
    }

    if (*beta == T(0))
    {
        for (len_type i = 0;i < MR;i++)
            for (len_type j = 0;j < NR;j++)
                p_c[i*rs_c + j*cs_c] = (*alpha)*p_ab[i*NR + j];
    }
    else
    {
        for (len_type i = 0;i < MR;i++)
            for (len_type j = 0;j < NR;j++)
                p_c[i*rs_c + j*cs_c] = (*alpha)*p_ab[i*NR + j] +
                                       (*beta)*p_c[i*rs_c + j*cs_c];
    }
}

}

#endif
