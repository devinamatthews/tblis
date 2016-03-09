#ifndef _TBLIS_IMPL_BLAS_HPP_
#define _TBLIS_IMPL_BLAS_HPP_

#include "impl/tensor_impl.hpp"

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_mult_blas(T alpha, const Tensor<T>& A, const std::string& idx_A,
                              const Tensor<T>& B, const std::string& idx_B,
                     T  beta,       Tensor<T>& C, const std::string& idx_C);

template <typename T>
int tensor_contract_blas(T alpha, const Tensor<T>& A, const std::string& idx_A,
                                  const Tensor<T>& B, const std::string& idx_B,
                         T  beta,       Tensor<T>& C, const std::string& idx_C);

template <typename T>
int tensor_weight_blas(T alpha, const Tensor<T>& A, const std::string& idx_A,
                                const Tensor<T>& B, const std::string& idx_B,
                       T  beta,       Tensor<T>& C, const std::string& idx_C);

template <typename T>
int tensor_outer_prod_blas(T alpha, const Tensor<T>& A, const std::string& idx_A,
                                    const Tensor<T>& B, const std::string& idx_B,
                           T  beta,       Tensor<T>& C, const std::string& idx_C);

template <typename T>
int tensor_sum_blas(T alpha, const Tensor<T>& A, const std::string& idx_A,
                    T  beta,       Tensor<T>& B, const std::string& idx_B);

template <typename T>
int tensor_trace_blas(T alpha, const Tensor<T>& A, const std::string& idx_A,
                      T  beta,       Tensor<T>& B, const std::string& idx_B);

template <typename T>
int tensor_replicate_blas(T alpha, const Tensor<T>& A, const std::string& idx_A,
                          T  beta,       Tensor<T>& B, const std::string& idx_B);

template <typename T>
int tensor_transpose_blas(T alpha, const Tensor<T>& A, const std::string& idx_A,
                          T  beta,       Tensor<T>& B, const std::string& idx_B);

template <typename T>
int tensor_dot_blas(const Tensor<T>& A, const std::string& idx_A,
                    const Tensor<T>& B, const std::string& idx_B, T& val);

template <typename T>
int tensor_scale_blas(T alpha, Tensor<T>& A, const std::string& idx_A);

template <typename T>
int tensor_reduce_blas(reduce_t op, const Tensor<T>& A, const std::string& idx_A, T& val, dim_t& idx);

}
}

#endif
