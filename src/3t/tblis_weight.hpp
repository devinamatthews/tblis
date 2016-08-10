#ifndef _TBLIS_WEIGHT_HPP_
#define _TBLIS_WEIGHT_HPP_

#include "tblis_marray.hpp"

namespace tblis
{

/*******************************************************************************
 *
 * Weight a tensor by a second and sum onto a third
 *
 * The general form for a weighting is ab...ef... * ef...cd... -> ab...cd...ef...
 * with no indices being summed over. Indices may be transposed in any tensor.
 * Any index group may be empty (in the case that ef... is empty, this reduces
 * to an outer product).
 *
 ******************************************************************************/

template <typename T>
int tensor_weight(T alpha, const_tensor_view<T> A, std::string idx_A,
                           const_tensor_view<T> B, std::string idx_B,
                  T  beta,       tensor_view<T> C, std::string idx_C);

template <typename T>
int tensor_weight_ref(const std::vector<idx_type>& len_AC,
                      const std::vector<idx_type>& len_BC,
                      const std::vector<idx_type>& len_ABC,
                      T alpha, const T* A, const std::vector<stride_type>& stride_A_AC,
                                           const std::vector<stride_type>& stride_A_ABC,
                               const T* B, const std::vector<stride_type>& stride_B_BC,
                                           const std::vector<stride_type>& stride_B_ABC,
                      T  beta,       T* C, const std::vector<stride_type>& stride_C_AC,
                                           const std::vector<stride_type>& stride_C_BC,
                                           const std::vector<stride_type>& stride_C_ABC);

template <typename T>
int tensor_weight_blas(const std::vector<idx_type>& len_AC,
                       const std::vector<idx_type>& len_BC,
                       const std::vector<idx_type>& len_ABC,
                       T alpha, const T* A, const std::vector<stride_type>& stride_A_AC,
                                            const std::vector<stride_type>& stride_A_ABC,
                                const T* B, const std::vector<stride_type>& stride_B_BC,
                                            const std::vector<stride_type>& stride_B_ABC,
                       T  beta,       T* C, const std::vector<stride_type>& stride_C_AC,
                                            const std::vector<stride_type>& stride_C_BC,
                                            const std::vector<stride_type>& stride_C_ABC);

}

#endif
