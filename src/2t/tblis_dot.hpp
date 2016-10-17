#ifndef _TBLIS_DOT_HPP_
#define _TBLIS_DOT_HPP_

#include "../util/marray.hpp"

namespace tblis
{

/*******************************************************************************
 *
 * Return the dot product of two tensors
 *
 ******************************************************************************/

template <typename T>
T tensor_dot(const_tensor_view<T> A, std::string idx_A,
             const_tensor_view<T> B, std::string idx_B)
{
    T val;
    int ret = tensor_dot(A, idx_A, B, idx_B, val);
    return val;
}

template <typename T>
int tensor_dot(const_tensor_view<T> A, std::string idx_A,
               const_tensor_view<T> B, std::string idx_B, T& val);

template <typename T>
int tensor_dot_ref(const std::vector<len_type>& len_AB,
                   const T* A, const std::vector<stride_type>& stride_A_AB,
                   const T* B, const std::vector<stride_type>& stride_B_AB,
                   T& val);

}

#endif
