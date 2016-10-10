#ifndef _TBLIS_REPLICATE_HPP_
#define _TBLIS_REPLICATE_HPP_

#include "tblis_marray.hpp"

namespace tblis
{

/*******************************************************************************
 *
 * Replicate a tensor and sum onto a second
 *
 * The general form for a replication operation is ab... -> ab...c*d*... where
 * c* denotes the index c appearing one or more times. Any indices may be
 * transposed.
 *
 ******************************************************************************/

template <typename T>
int tensor_replicate(T alpha, const_tensor_view<T> A, std::string idx_A,
                     T  beta,       tensor_view<T> B, std::string idx_B);

template <typename T>
int tensor_replicate_ref(const std::vector<len_type>& len_B,
                         const std::vector<len_type>& len_AB,
                         T alpha, const T* A, const std::vector<stride_type>& stride_A_AB,
                         T  beta,       T* B, const std::vector<stride_type>& stride_B_B,
                                              const std::vector<stride_type>& stride_B_AB);

}

#endif
