#ifndef _TBLIS_IMPL_BLIS_HPP_
#define _TBLIS_IMPL_BLIS_HPP_

#include "impl/tensor_impl.hpp"

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_mult_blis(T alpha, const Tensor<T>& A, const std::string& idx_A,
                              const Tensor<T>& B, const std::string& idx_B,
                     T  beta,       Tensor<T>& C, const std::string& idx_C);

template <typename T>
int tensor_contract_blis(T alpha, const Tensor<T>& A, const std::string& idx_A,
                                  const Tensor<T>& B, const std::string& idx_B,
                         T  beta,       Tensor<T>& C, const std::string& idx_C);

template <typename T>
int tensor_weight_blis(T alpha, const Tensor<T>& A, const std::string& idx_A,
                                const Tensor<T>& B, const std::string& idx_B,
                       T  beta,       Tensor<T>& C, const std::string& idx_C);

template <typename T>
int tensor_outer_prod_blis(T alpha, const Tensor<T>& A, const std::string& idx_A,
                                    const Tensor<T>& B, const std::string& idx_B,
                           T  beta,       Tensor<T>& C, const std::string& idx_C);

template <typename T>
int tensor_sum_blis(T alpha, const Tensor<T>& A, const std::string& idx_A,
                    T  beta,       Tensor<T>& B, const std::string& idx_B);

template <typename T>
int tensor_trace_blis(T alpha, const Tensor<T>& A, const std::string& idx_A,
                      T  beta,       Tensor<T>& B, const std::string& idx_B);

template <typename T>
int tensor_replicate_blis(T alpha, const Tensor<T>& A, const std::string& idx_A,
                          T  beta,       Tensor<T>& B, const std::string& idx_B);

template <typename T>
int tensor_transpose_blis(T alpha, const Tensor<T>& A, const std::string& idx_A,
                          T  beta,       Tensor<T>& B, const std::string& idx_B);

template <typename T>
int tensor_dot_blis(const Tensor<T>& A, const std::string& idx_A,
                    const Tensor<T>& B, const std::string& idx_B, T& val);

template <typename T>
int tensor_scale_blis(T alpha, Tensor<T>& A, const std::string& idx_A);

template <typename T>
int tensor_reduce_blis(reduce_t op, const Tensor<T>& A, const std::string& idx_A, T& val, dim_t& idx);

}
}

#endif
