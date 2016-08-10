#ifndef _TBLIS_SUM_HPP_
#define _TBLIS_SUM_HPP_

#include "tblis_marray.hpp"

namespace tblis
{

template <typename T>
int tensor_sum(T alpha, const_tensor_view<T> A, std::string idx_A,
               T  beta,       tensor_view<T> B, std::string idx_B);

template <typename T>
int tensor_sum_ref(const std::vector<idx_type>& len_A,
                   const std::vector<idx_type>& len_B,
                   const std::vector<idx_type>& len_AB,
                   T alpha, const T* A, const std::vector<stride_type>& stride_A_A,
                                        const std::vector<stride_type>& stride_A_AB,
                   T  beta,       T* B, const std::vector<stride_type>& stride_B_B,
                                        const std::vector<stride_type>& stride_B_AB);

}

#endif
