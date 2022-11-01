#ifndef TBLIS_KERNELS_1V_SET_HPP
#define TBLIS_KERNELS_1V_SET_HPP 1

#include <tblis/internal/types.hpp>

namespace tblis
{

template <typename Config, typename T>
void set_ukr_def(len_type n,
                 const void* alpha_, void* A_, stride_type inc_A)
{
    T alpha = *static_cast<const T*>(alpha_);

    T* TBLIS_RESTRICT A = static_cast<T*>(A_);

    if (inc_A == 1)
    {
        for (len_type i = 0;i < n;i++) A[i] = alpha;
    }
    else
    {
        for (len_type i = 0;i < n;i++) A[i*inc_A] = alpha;
    }
}

}

#endif //TBLIS_KERNELS_1V_SET_HPP
