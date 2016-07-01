#ifndef _TBLIS_NORMFM_HPP_
#define _TBLIS_NORMFM_HPP_

#include "tblis.hpp"

namespace tblis
{

template <typename T>
void tblis_normfm(const_matrix_view<T> A, T& norm);

template <typename T>
void tblis_normfm(const_scatter_matrix_view<T> A, T& norm);

template <typename T>
T tblis_normfm(const_matrix_view<T> A);

template <typename T>
T tblis_normfm(const_scatter_matrix_view<T> A);

}

#endif
