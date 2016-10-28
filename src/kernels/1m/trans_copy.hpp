#ifndef _TBLIS_KERNELS_1M_TRANS_COPY_HPP_
#define _TBLIS_KERNELS_1M_TRANS_COPY_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "util/macros.h"

namespace tblis
{

template <typename T>
using trans_copy_ukr_t =
    void (*)(len_type m, len_type n,
             T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
                                         T* B, stride_type rs_B, stride_type cs_B);

template <typename Config, typename T>
void trans_copy_ukr_def(len_type m, len_type n,
                        T alpha, bool conj_A, const T* TBLIS_RESTRICT A, stride_type rs_A, stride_type cs_A,
                                                    T* TBLIS_RESTRICT B, stride_type rs_B, stride_type cs_B)
{
    constexpr len_type MR = Config::template trans_mr<T>::def;
    constexpr len_type NR = Config::template trans_nr<T>::def;

    TBLIS_SPECIAL_CASE(alpha == T(1),
    TBLIS_SPECIAL_CASE(is_complex<T>::value && conj_A,
    TBLIS_SPECIAL_CASE(m == MR && n == NR && rs_B == 1 && cs_A == 1,
    {
         for (len_type i = 0;i < m;i++)
         {
            for (len_type j = 0;j < n;j++)
            {
                B[j*rs_B + i*cs_B] = alpha*conj(conj_A, A[i*rs_A + j*cs_A]);
            }
        }
    }
    )))

}

}

#endif
