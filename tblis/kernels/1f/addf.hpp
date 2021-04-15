#ifndef TBLIS_KERNELS_1F_ADDF_HPP
#define TBLIS_KERNELS_1F_ADDF_HPP 1

#include <tblis/internal/types.hpp>

namespace tblis
{

template <typename Config, typename T>
void addf_sum_ukr_def(len_type m, len_type n,
                      const void* alpha_, bool conj_A, const void** A_, stride_type inc_A,
                                          bool conj_B, const void*  B_, stride_type inc_B,
                      const void*  beta_, bool conj_C,       void*  C_, stride_type inc_C)
{
    constexpr len_type NF = Config::template addf_nf<T>::def;

    T alpha = *reinterpret_cast<const T*>(alpha_);
    T beta  = *reinterpret_cast<const T*>(beta_ );

    const T* TBLIS_RESTRICT * A = reinterpret_cast<const T**>(A_);
    const T* TBLIS_RESTRICT   B = reinterpret_cast<const T* >(B_);
          T* TBLIS_RESTRICT   C = reinterpret_cast<      T* >(C_);

    T alpha_B[NF];

    for (len_type i = 0;i < n;i++)
        alpha_B[i] = alpha*conj(conj_B, B[i*inc_B]);

    if (n == NF)
    {
        if (beta == T(0))
        {
            if (is_complex<T>::value && conj_A)
            {
                if (inc_A == 1 && inc_C == 1)
                {
                    #pragma omp simd
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i] = T(0);

                        for (len_type j = 0;j < n;j++)
                            C[i] += alpha_B[j]*conj(A[j][i]);
                    }
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i*inc_C] = T(0);

                        for (len_type j = 0;j < n;j++)
                            C[i*inc_C] += alpha_B[j]*conj(A[j][i*inc_A]);
                    }
                }
            }
            else
            {
                if (inc_A == 1 && inc_C == 1)
                {
                    #pragma omp simd
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i] = T(0);

                        for (len_type j = 0;j < n;j++)
                            C[i] += alpha_B[j]*A[j][i];
                    }
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i*inc_C] = T(0);

                        for (len_type j = 0;j < n;j++)
                            C[i*inc_C] += alpha_B[j]*A[j][i*inc_A];
                    }
                }
            }
        }
        else
        {
            if (is_complex<T>::value && conj_A && conj_C)
            {
                if (inc_A == 1 && inc_C == 1)
                {
                    #pragma omp simd
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i] = beta*conj(C[i]);

                        for (len_type j = 0;j < n;j++)
                            C[i] += alpha_B[j]*conj(A[j][i]);
                    }
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i*inc_C] = beta*conj(C[i*inc_C]);

                        for (len_type j = 0;j < n;j++)
                            C[i*inc_C] += alpha_B[j]*conj(A[j][i*inc_A]);
                    }
                }
            }
            else if (is_complex<T>::value && conj_C)
            {
                if (inc_A == 1 && inc_C == 1)
                {
                    #pragma omp simd
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i] = beta*conj(C[i]);

                        for (len_type j = 0;j < n;j++)
                            C[i] += alpha_B[j]*A[j][i];
                    }
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i*inc_C] = beta*conj(C[i*inc_C]);

                        for (len_type j = 0;j < n;j++)
                            C[i*inc_C] += alpha_B[j]*A[j][i*inc_A];
                    }
                }
            }
            else if (is_complex<T>::value && conj_A)
            {
                if (inc_A == 1 && inc_C == 1)
                {
                    #pragma omp simd
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i] = beta*C[i];

                        for (len_type j = 0;j < n;j++)
                            C[i] += alpha_B[j]*conj(A[j][i]);
                    }
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i*inc_C] = beta*C[i*inc_C];

                        for (len_type j = 0;j < n;j++)
                            C[i*inc_C] += alpha_B[j]*conj(A[j][i*inc_A]);
                    }
                }
            }
            else
            {
                if (inc_A == 1 && inc_C == 1)
                {
                    #pragma omp simd
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i] = beta*C[i];

                        for (len_type j = 0;j < n;j++)
                            C[i] += alpha_B[j]*A[j][i];
                    }
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                    {
                        C[i*inc_C] = beta*C[i*inc_C];

                        for (len_type j = 0;j < n;j++)
                            C[i*inc_C] += alpha_B[j]*A[j][i*inc_A];
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
                C[i*inc_C] += alpha_B[j]*conj(conj_A, A[j][i*inc_A]);
    }
}

using addf_rep_ukr_t =
    void (*)(len_type m, len_type n,
             const void* alpha, bool conj_A, const void*  A, stride_type inc_A,
                                bool conj_B, const void*  B, stride_type inc_B,
             const void*  beta, bool conj_C,       void** C, stride_type inc_C);

template <typename Config, typename T>
void addf_rep_ukr_def(len_type m, len_type n,
                      const void* alpha_, bool conj_A, const void*  A_, stride_type inc_A,
                                          bool conj_B, const void*  B_, stride_type inc_B,
                      const void*  beta_, bool conj_C,       void** C_, stride_type inc_C)
{
    constexpr len_type NF = Config::template addf_nf<T>::def;

    T alpha = *reinterpret_cast<const T*>(alpha_);
    T beta  = *reinterpret_cast<const T*>(beta_ );

    const T* TBLIS_RESTRICT   A = reinterpret_cast<const T* >(A_);
    const T* TBLIS_RESTRICT   B = reinterpret_cast<const T* >(B_);
          T* TBLIS_RESTRICT * C = reinterpret_cast<      T**>(C_);

    T alpha_B[NF];

    for (len_type i = 0;i < n;i++)
        alpha_B[i] = alpha*conj(conj_B, B[i*inc_B]);

    if (n == NF)
    {
        if (beta == T(0))
        {
            if (is_complex<T>::value && conj_A)
            {
                if (inc_A == 1 && inc_C == 1)
                {
                    #pragma omp simd
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            C[j][i] = alpha_B[j]*conj(A[i]);
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            C[j][i*inc_C] = alpha_B[j]*conj(A[i*inc_A]);
                }
            }
            else
            {
                if (inc_A == 1 && inc_C == 1)
                {
                    #pragma omp simd
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            C[j][i] = alpha_B[j]*A[i];
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            C[j][i*inc_C] = alpha_B[j]*A[i*inc_A];
                }
            }
        }
        else
        {
            if (is_complex<T>::value && conj_A && conj_C)
            {
                if (inc_A == 1 && inc_C == 1)
                {
                    #pragma omp simd
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            C[j][i] = alpha_B[j]*conj(A[i]) + beta*conj(C[j][i]);
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            C[j][i*inc_C] = alpha_B[j]*conj(A[i*inc_A]) + beta*conj(C[j][i*inc_C]);
                }
            }
            else if (is_complex<T>::value && conj_C)
            {
                if (inc_A == 1 && inc_C == 1)
                {
                    #pragma omp simd
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            C[j][i] = alpha_B[j]*A[i] + beta*conj(C[j][i]);
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            C[j][i*inc_C] = alpha_B[j]*A[i*inc_A] + beta*conj(C[j][i*inc_C]);
                }
            }
            else if (is_complex<T>::value && conj_A)
            {
                if (inc_A == 1 && inc_C == 1)
                {
                    #pragma omp simd
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            C[j][i] = alpha_B[j]*conj(A[i]) + beta*C[j][i];
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            C[j][i*inc_C] = alpha_B[j]*conj(A[i*inc_A]) + beta*C[j][i*inc_C];
                }
            }
            else
            {
                if (inc_A == 1 && inc_C == 1)
                {
                    #pragma omp simd
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            C[j][i] = alpha_B[j]*A[i] + beta*C[j][i];
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            C[j][i*inc_C] = alpha_B[j]*A[i*inc_A] + beta*C[j][i*inc_C];
                }
            }
        }
    }
    else
    {
        if (beta == T(0))
        {
            for (len_type j = 0;j < n;j++)
                for (len_type i = 0;i < m;i++)
                    C[j][i*inc_C] = alpha_B[j]*conj(conj_A, A[i*inc_A]);
        }
        else
        {
            for (len_type j = 0;j < n;j++)
                for (len_type i = 0;i < m;i++)
                    C[j][i*inc_C] = alpha_B[j]*conj(conj_A, A[i*inc_A]) + beta*conj(conj_C, C[j][i*inc_C]);
        }
    }
}

}

#endif //TBLIS_KERNELS_1F_ADDF_HPP
