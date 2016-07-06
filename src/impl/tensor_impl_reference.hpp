#ifndef _TBLIS_IMPL_REFERENCE_HPP_
#define _TBLIS_IMPL_REFERENCE_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_mult_reference(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                   const const_tensor_view<T>& B, const std::string& idx_B,
                          T  beta, const       tensor_view<T>& C, const std::string& idx_C);

template <typename T>
int tensor_contract_reference(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                       const const_tensor_view<T>& B, const std::string& idx_B,
                              T  beta, const       tensor_view<T>& C, const std::string& idx_C);

template <typename T>
int tensor_weight_reference(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                     const const_tensor_view<T>& B, const std::string& idx_B,
                            T  beta, const       tensor_view<T>& C, const std::string& idx_C);

template <typename T>
int tensor_outer_prod_reference(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                         const const_tensor_view<T>& B, const std::string& idx_B,
                                T  beta, const       tensor_view<T>& C, const std::string& idx_C);

template <typename T>
int tensor_sum_reference(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                         T  beta, const       tensor_view<T>& B, const std::string& idx_B);

template <typename T>
int tensor_trace_reference(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                           T  beta, const       tensor_view<T>& B, const std::string& idx_B);

template <typename T>
int tensor_replicate_reference(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                               T  beta, const       tensor_view<T>& B, const std::string& idx_B);

template <typename T>
int tensor_transpose_reference(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                               T  beta, const       tensor_view<T>& B, const std::string& idx_B);

template <typename T>
int tensor_dot_reference(const const_tensor_view<T>& A, const std::string& idx_A,
                         const const_tensor_view<T>& B, const std::string& idx_B, T& val);

template <typename T>
int tensor_scale_reference(T alpha, const tensor_view<T>& A, const std::string& idx_A);

template <typename T>
int tensor_reduce_reference(reduce_t op, const const_tensor_view<T>& A, const std::string& idx_A, T& val, stride_type& idx);

}
}

#endif
