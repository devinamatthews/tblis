#ifndef TBLIS_KERNELS_1V_MULT_HPP
#define TBLIS_KERNELS_1V_MULT_HPP 1

#include <tblis/internal/types.hpp>

namespace tblis
{

template <typename Config, typename T>
void mult_ukr_def(len_type n,
                  const void* alpha_, bool conj_A, const void* A_, stride_type inc_A,
                                      bool conj_B, const void* B_, stride_type inc_B,
                  const void*  beta_, bool conj_C,       void* C_, stride_type inc_C)
{
    T alpha = *static_cast<const T*>(alpha_);
    T beta  = *static_cast<const T*>(beta_ );

    const T* TBLIS_RESTRICT A = static_cast<const T*>(A_);
    const T* TBLIS_RESTRICT B = static_cast<const T*>(B_);
          T* TBLIS_RESTRICT C = static_cast<      T*>(C_);

    if (is_complex<T>::value && conj_B && !conj_A)
    {
        std::swap(conj_A, conj_B);
        std::swap(A, B);
        std::swap(inc_A, inc_B);
    }

    if (beta == T(0))
    {
        if (is_complex<T>::value && conj_A && conj_B)
        {
            if (inc_A == 1 && inc_B == 1 && inc_C == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    C[i] = alpha*conj(A[i])*conj(B[i]);
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    C[i*inc_C] = alpha*conj(A[i*inc_A])*conj(B[i*inc_B]);
            }
        }
        else if (is_complex<T>::value && conj_A)
        {
            if (inc_A == 1 && inc_B == 1 && inc_C == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    C[i] = alpha*conj(A[i])*B[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    C[i*inc_C] = alpha*conj(A[i*inc_A])*B[i*inc_B];
            }
        }
        else
        {
            if (inc_A == 1 && inc_B == 1 && inc_C == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    C[i] = alpha*A[i]*B[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    C[i*inc_C] = alpha*A[i*inc_A]*B[i*inc_B];
            }
        }
    }
    else
    {
        if (is_complex<T>::value && conj_A && conj_B && conj_C)
        {
            if (inc_A == 1 && inc_B == 1 && inc_C == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    C[i] = alpha*conj(A[i])*conj(B[i]) + beta*conj(C[i]);
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    C[i*inc_C] = alpha*conj(A[i*inc_A])*conj(B[i*inc_B]) +
                                  beta*conj(C[i*inc_C]);
            }
        }
        else if (is_complex<T>::value && conj_A && conj_C)
        {
            if (inc_A == 1 && inc_B == 1 && inc_C == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    C[i] = alpha*conj(A[i])*B[i] + beta*conj(C[i]);
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    C[i*inc_C] = alpha*conj(A[i*inc_A])*B[i*inc_B] +
                                  beta*conj(C[i*inc_C]);
            }
        }
        else if (is_complex<T>::value && conj_C)
        {
            if (inc_A == 1 && inc_B == 1 && inc_C == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    C[i] = alpha*A[i]*B[i] + beta*conj(C[i]);
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    C[i*inc_C] = alpha*A[i*inc_A]*B[i*inc_B] +
                                  beta*conj(C[i*inc_C]);
            }
        }
        else if (is_complex<T>::value && conj_A && conj_B)
        {
            if (inc_A == 1 && inc_B == 1 && inc_C == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    C[i] = alpha*conj(A[i])*conj(B[i]) + beta*C[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    C[i*inc_C] = alpha*conj(A[i*inc_A])*conj(B[i*inc_B]) +
                                  beta*C[i*inc_C];
            }
        }
        else if (is_complex<T>::value && conj_A)
        {
            if (inc_A == 1 && inc_B == 1 && inc_C == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    C[i] = alpha*conj(A[i])*B[i] + beta*C[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    C[i*inc_C] = alpha*conj(A[i*inc_A])*B[i*inc_B] +
                                  beta*C[i*inc_C];
            }
        }
        else
        {
            if (inc_A == 1 && inc_B == 1 && inc_C == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    C[i] = alpha*A[i]*B[i] + beta*C[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    C[i*inc_C] = alpha*A[i*inc_A]*B[i*inc_B] + beta*C[i*inc_C];
            }
        }
    }
}

}

#endif //TBLIS_KERNELS_1V_MULT_HPP
