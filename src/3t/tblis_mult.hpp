#ifndef _TBLIS_MULT_HPP_
#define _TBLIS_MULT_HPP_

#include "tblis_marray.hpp"

namespace tblis
{

/*******************************************************************************
 *
 * Multiply two tensors together and sum onto a third
 *
 * This form generalizes contraction and weighting with the unary operations
 * trace, transpose, and replicate. Note that the binary contraction operation
 * is similar in form to the unary trace operation, while the binary weighting
 * operation is similar in form to the unary diagonal operation. Any combination
 * of these operations may be performed.
 *
 ******************************************************************************/

template <typename T>
int tensor_mult(T alpha, const_tensor_view<T> A, std::string idx_A,
                         const_tensor_view<T> B, std::string idx_B,
                T  beta,       tensor_view<T> C, std::string idx_C);

template <typename T>
int tensor_mult_ref(const std::vector<len_type>& len_A,
                    const std::vector<len_type>& len_B,
                    const std::vector<len_type>& len_C,
                    const std::vector<len_type>& len_AB,
                    const std::vector<len_type>& len_AC,
                    const std::vector<len_type>& len_BC,
                    const std::vector<len_type>& len_ABC,
                    T alpha, const T* A, const std::vector<stride_type>& stride_A_A,
                                         const std::vector<stride_type>& stride_A_AB,
                                         const std::vector<stride_type>& stride_A_AC,
                                         const std::vector<stride_type>& stride_A_ABC,
                             const T* B, const std::vector<stride_type>& stride_B_B,
                                         const std::vector<stride_type>& stride_B_AB,
                                         const std::vector<stride_type>& stride_B_BC,
                                         const std::vector<stride_type>& stride_B_ABC,
                    T  beta,       T* C, const std::vector<stride_type>& stride_C_C,
                                         const std::vector<stride_type>& stride_C_AC,
                                         const std::vector<stride_type>& stride_C_BC,
                                         const std::vector<stride_type>& stride_C_ABC);

template <typename T>
int tensor_mult_blas(const std::vector<len_type>& len_A,
                     const std::vector<len_type>& len_B,
                     const std::vector<len_type>& len_C,
                     const std::vector<len_type>& len_AB,
                     const std::vector<len_type>& len_AC,
                     const std::vector<len_type>& len_BC,
                     const std::vector<len_type>& len_ABC,
                     T alpha, const T* A, const std::vector<stride_type>& stride_A_A,
                                          const std::vector<stride_type>& stride_A_AB,
                                          const std::vector<stride_type>& stride_A_AC,
                                          const std::vector<stride_type>& stride_A_ABC,
                              const T* B, const std::vector<stride_type>& stride_B_B,
                                          const std::vector<stride_type>& stride_B_AB,
                                          const std::vector<stride_type>& stride_B_BC,
                                          const std::vector<stride_type>& stride_B_ABC,
                     T  beta,       T* C, const std::vector<stride_type>& stride_C_C,
                                          const std::vector<stride_type>& stride_C_AC,
                                          const std::vector<stride_type>& stride_C_BC,
                                          const std::vector<stride_type>& stride_C_ABC);

}

#endif
