#ifndef TBLIS_KERNELS_1F_DOTF_HPP
#define TBLIS_KERNELS_1F_DOTF_HPP 1

#include <tblis/internal/types.hpp>

namespace tblis
{

template <typename Config, typename T>
void dotf_ukr_def(len_type m, len_type n,
                  const void* alpha_, bool conj_A, const void* A_, stride_type rs_A, stride_type cs_A,
                                      bool conj_B, const void* B_, stride_type inc_B,
                  const void*  beta_, bool conj_C,       void* C_, stride_type inc_C)
{
    constexpr len_type NF = Config::template dotf_nf<T>::def;

    T alpha = *static_cast<const T*>(alpha_);
    T beta  = *static_cast<const T*>(beta_ );

    const T* TBLIS_RESTRICT A = static_cast<const T*>(A_);
    const T* TBLIS_RESTRICT B = static_cast<const T*>(B_);
          T* TBLIS_RESTRICT C = static_cast<      T*>(C_);

    T AB[NF] = {};

    if (conj_A) conj_B = !conj_B;

    if (cs_A == 1 && inc_B == 1)
    {
        for (len_type i = 0;i < m;i++)
            //icpc 2021 has problems
            //#pragma omp simd
            for (len_type j = 0;j < n;j++)
                AB[i] += A[i*rs_A + j]*conj(conj_B, B[j]);
    }
    else
    {
        for (len_type i = 0;i < m;i++)
            for (len_type j = 0;j < n;j++)
                AB[i] += A[i*rs_A + j*cs_A]*conj(conj_B, B[j*inc_B]);
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

#endif //TBLIS_KERNELS_1F_DOTF_HPP
