#ifndef _TBLIS_NORMFM_HPP_
#define _TBLIS_NORMFM_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T, typename MatrixA>
void tblis_normfm(const MatrixA& A, T& norm);

template <typename MatrixA>
typename MatrixA::type tblis_normfm(const MatrixA& A);

}
}

#endif
