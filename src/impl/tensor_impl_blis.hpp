#ifndef _TBLIS_IMPL_BLIS_HPP_
#define _TBLIS_IMPL_BLIS_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_contract_blis(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                  const const_tensor_view<T>& B, const std::string& idx_B,
                         T  beta, const       tensor_view<T>& C, const std::string& idx_C);

}
}

#endif
