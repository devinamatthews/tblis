#ifndef _TBLIS_TRANSPOSE_HPP_
#define _TBLIS_TRANSPOSE_HPP_

#include "../util/marray.hpp"

namespace tblis
{

/*******************************************************************************
 *
 * Transpose a tensor and sum onto a second
 *
 * The general form for a transposition operation is ab... -> P(ab...) where P
 * is some permutation. Transposition may change the order in which the elements
 * of the tensor are physically stored.
 *
 ******************************************************************************/

template <typename T>
int tensor_transpose(T alpha, const_tensor_view<T> A, std::string idx_A,
                     T  beta,       tensor_view<T> B, std::string idx_B);

template <typename T>
int tensor_transpose_ref(const std::vector<len_type>& len_AB,
                         T alpha, const T* A, const std::vector<stride_type>& stride_A_AB,
                         T  beta,       T* B, const std::vector<stride_type>& stride_B_AB);

}

#endif
