#ifndef _TBLIS_KERNELS_1V_REDUCE_HPP_
#define _TBLIS_KERNELS_1V_REDUCE_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "util/macros.h"

namespace tblis
{

template <typename T>
using reduce_ukr_t =
    void (*)(reduce_t op, len_type n,
             const T* A, stride_type inc_A, T& value, len_type& idx);

template <typename Config, typename T>
void reduce_ukr_def(reduce_t op, len_type n,
                    const T* A, stride_type inc_A, T& value, len_type& idx)
{
    if (op == REDUCE_SUM)
    {
        if (inc_A == 1)
        {
            #pragma omp simd
            for (len_type i = 0;i < n;i++) value += A[i];
        }
        else
        {
            for (len_type i = 0;i < n;i++) value += A[i*inc_A];
        }
    }
    else if (op == REDUCE_SUM_ABS)
    {
        if (inc_A == 1)
        {
            #pragma omp simd
            for (len_type i = 0;i < n;i++) value += std::abs(A[i]);
        }
        else
        {
            for (len_type i = 0;i < n;i++) value += std::abs(A[i*inc_A]);
        }
    }
    else if (op == REDUCE_MAX)
    {
        for (len_type i = 0;i < n;i++)
        {
            if (A[i*inc_A] > value)
            {
                value = A[i*inc_A];
                idx = i*inc_A;
            }
        }
    }
    else if (op == REDUCE_MAX_ABS)
    {
        for (len_type i = 0;i < n;i++)
        {
            if (std::abs(A[i*inc_A]) > value)
            {
                value = std::abs(A[i*inc_A]);
                idx = i*inc_A;
            }
        }
    }
    else if (op == REDUCE_MIN)
    {
        for (len_type i = 0;i < n;i++)
        {
            if (A[i*inc_A] < value)
            {
                value = A[i*inc_A];
                idx = i*inc_A;
            }
        }
    }
    else if (op == REDUCE_MIN_ABS)
    {
        for (len_type i = 0;i < n;i++)
        {
            if (std::abs(A[i*inc_A]) < value)
            {
                value = std::abs(A[i*inc_A]);
                idx = i*inc_A;
            }
        }
    }
    else if (op == REDUCE_NORM_2)
    {
        if (inc_A == 1)
        {
            #pragma omp simd
            for (len_type i = 0;i < n;i++) value += norm2(A[i]);
        }
        else
        {
            for (len_type i = 0;i < n;i++) value += norm2(A[i*inc_A]);
        }
    }
}

}

#endif
