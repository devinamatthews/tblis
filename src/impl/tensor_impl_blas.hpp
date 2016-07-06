#ifndef _TBLIS_IMPL_BLAS_HPP_
#define _TBLIS_IMPL_BLAS_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_mult_blas(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                              const const_tensor_view<T>& B, const std::string& idx_B,
                     T  beta, const       tensor_view<T>& C, const std::string& idx_C);

template <typename T>
int tensor_contract_blas(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                  const const_tensor_view<T>& B, const std::string& idx_B,
                         T  beta, const       tensor_view<T>& C, const std::string& idx_C);

template <typename T>
int tensor_weight_blas(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                const const_tensor_view<T>& B, const std::string& idx_B,
                       T  beta, const       tensor_view<T>& C, const std::string& idx_C);

template <typename T>
int tensor_outer_prod_blas(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                    const const_tensor_view<T>& B, const std::string& idx_B,
                           T  beta, const       tensor_view<T>& C, const std::string& idx_C);

}
}

#endif
