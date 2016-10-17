#ifndef _TBLIS_OUTER_PROD_HPP_
#define _TBLIS_OUTER_PROD_HPP_

#include "../util/marray.hpp"

namespace tblis
{

/*******************************************************************************
 *
 * Sum the outer product of two tensors onto a third
 *
 * The general form for an outer product is ab... * cd... -> ab...cd... with no
 * indices being summed over. Indices may be transposed in any tensor.
 *
 ******************************************************************************/

template <typename T>
int tensor_outer_prod(T alpha, const_tensor_view<T> A, std::string idx_A,
                               const_tensor_view<T> B, std::string idx_B,
                      T  beta,       tensor_view<T> C, std::string idx_C);

template <typename T>
int tensor_outer_prod_ref(const std::vector<len_type>& len_AC,
                          const std::vector<len_type>& len_BC,
                          T alpha, const T* A, const std::vector<stride_type>& stride_A_AC,
                                   const T* B, const std::vector<stride_type>& stride_B_BC,
                          T  beta,       T* C, const std::vector<stride_type>& stride_C_AC,
                                               const std::vector<stride_type>& stride_C_BC);

template <typename T>
int tensor_outer_prod_blas(const std::vector<len_type>& len_AC,
                           const std::vector<len_type>& len_BC,
                           T alpha, const T* A, const std::vector<stride_type>& stride_A_AC,
                                    const T* B, const std::vector<stride_type>& stride_B_BC,
                           T  beta,       T* C, const std::vector<stride_type>& stride_C_AC,
                                                const std::vector<stride_type>& stride_C_BC);

}

#endif
