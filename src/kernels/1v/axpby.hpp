#ifndef _TBLIS_KERNELS_1V_AXPBY_H_
#define _TBLIS_KERNELS_1V_AXPBY_H_

#include "util/thread.h"
#include "util/basic_types.h"
#include "util/assert.h"

namespace tblis
{

template <typename T>
using axpbyv_ker_t =
    void (*)(communicator& comm, bool conj_A, len_type n,
             T alpha, const T* A, stride_type inc_A,
             T  beta,       T* B, stride_type inc_B);

template <typename T>
void axpbyv_def(communicator& comm, bool conj_A, len_type n,
                T alpha, const T* A, stride_type inc_A,
                T  beta,       T* B, stride_type inc_B)
{
    len_type first, last;
    std::tie(first, last, std::ignore) = comm.distribute_over_threads(n);

    A += first*inc_A;
    B += first*inc_B;
    n = last-first;

    if (alpha == T(0))
    {
        if (beta == T(0))
        {
            // zero
            if (inc_A == 1 && inc_B == 1)
            {
                for (len_type i = 0;i < n;i++) B[i] = T();
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                {
                    *B = T();
                    B += inc_B;
                }
            }
        }
        else if (beta == T(1))
        {
            // no effect
        }
        else
        {
            // scal
            if (inc_A == 1 && inc_B == 1)
            {
                for (len_type i = 0;i < n;i++) B[i] *= beta;
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                {
                    *B *= beta;
                    B += inc_B;
                }
            }
        }
    }
    else if (alpha == T(1))
    {
        if (beta == T(0))
        {
            // copy
            if (inc_A == 1 && inc_B == 1)
            {
                for (len_type i = 0;i < n;i++) B[i] = A[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                {
                    *B = *A;
                    A += inc_A;
                    B += inc_B;
                }
            }
        }
        else if (beta == T(1))
        {
            // add
            if (inc_A == 1 && inc_B == 1)
            {
                for (len_type i = 0;i < n;i++) B[i] += A[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                {
                    *B += *A;
                    A += inc_A;
                    B += inc_B;
                }
            }
        }
        else
        {
            // xpby
            if (inc_A == 1 && inc_B == 1)
            {
                for (len_type i = 0;i < n;i++) B[i] = A[i] + beta*B[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                {
                    *B = *A + beta*(*B);
                    A += inc_A;
                    B += inc_B;
                }
            }
        }
    }
    else
    {
        if (beta == T(0))
        {
            // scal2
            if (inc_A == 1 && inc_B == 1)
            {
                for (len_type i = 0;i < n;i++) B[i] = alpha*A[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                {
                    *B = alpha*(*A);
                    A += inc_A;
                    B += inc_B;
                }
            }
        }
        else if (beta == T(1))
        {
            // axpy
            if (inc_A == 1 && inc_B == 1)
            {
                for (len_type i = 0;i < n;i++) B[i] += alpha*A[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                {
                    *B += alpha*(*A);
                    A += inc_A;
                    B += inc_B;
                }
            }
        }
        else
        {
            // axpby
            if (inc_A == 1 && inc_B == 1)
            {
                for (len_type i = 0;i < n;i++) B[i] = alpha*A[i] + beta*B[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                {
                    *B = alpha*(*A) + beta*(*B);
                    A += inc_A;
                    B += inc_B;
                }
            }
        }
    }
}

}

#endif
