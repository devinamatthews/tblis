#ifndef _TBLIS_CONTRACT_HPP_
#define _TBLIS_CONTRACT_HPP_

#include "tblis_marray.hpp"

namespace tblis
{

/*******************************************************************************
 *
 * Contract two tensors into a third
 *
 * The general form for a contraction is ab...ef... * ef...cd... -> ab...cd...
 * where the indices ef... will be summed over. Indices may be transposed in any
 * tensor. Any index group may be empty (in the case that ef... is empty, this
 * reduces to an outer product).
 *
 ******************************************************************************/

template <typename T>
int tensor_contract(T alpha, const_tensor_view<T> A, std::string idx_A,
                             const_tensor_view<T> B, std::string idx_B,
                    T  beta,       tensor_view<T> C, std::string idx_C);

template <typename T>
int tensor_contract_ref(const std::vector<len_type>& len_AB,
                        const std::vector<len_type>& len_AC,
                        const std::vector<len_type>& len_BC,
                        T alpha, const T* A, const std::vector<stride_type>& stride_A_AB,
                                             const std::vector<stride_type>& stride_A_AC,
                                 const T* B, const std::vector<stride_type>& stride_B_AB,
                                             const std::vector<stride_type>& stride_B_BC,
                        T  beta,       T* C, const std::vector<stride_type>& stride_C_AC,
                                             const std::vector<stride_type>& stride_C_BC);

template <typename T>
int tensor_contract_blas(const std::vector<len_type>& len_AB,
                         const std::vector<len_type>& len_AC,
                         const std::vector<len_type>& len_BC,
                         T alpha, const T* A, const std::vector<stride_type>& stride_A_AB,
                                              const std::vector<stride_type>& stride_A_AC,
                                  const T* B, const std::vector<stride_type>& stride_B_AB,
                                              const std::vector<stride_type>& stride_B_BC,
                         T  beta,       T* C, const std::vector<stride_type>& stride_C_AC,
                                              const std::vector<stride_type>& stride_C_BC);

template <typename T>
int tensor_contract_blis(const std::vector<len_type>& len_AB,
                         const std::vector<len_type>& len_AC,
                         const std::vector<len_type>& len_BC,
                         T alpha, const T* A, const std::vector<stride_type>& stride_A_AB,
                                              const std::vector<stride_type>& stride_A_AC,
                                  const T* B, const std::vector<stride_type>& stride_B_AB,
                                              const std::vector<stride_type>& stride_B_BC,
                         T  beta,       T* C, const std::vector<stride_type>& stride_C_AC,
                                              const std::vector<stride_type>& stride_C_BC);

}

#endif
