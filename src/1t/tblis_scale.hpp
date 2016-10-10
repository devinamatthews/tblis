#ifndef _TBLIS_SCALE_HPP_
#define _TBLIS_SCALE_HPP_

#include "tblis_marray.hpp"

namespace tblis
{

/*******************************************************************************
 *
 * Scale a tensor by a scalar
 *
 ******************************************************************************/

template <typename T>
int tensor_scale(T alpha, tensor_view<T> A, std::string idx_A);

template <typename T>
int tensor_scale_ref(const std::vector<len_type>& len_A,
                     T alpha, T* A, const std::vector<stride_type>& stride_A);

}

#endif
