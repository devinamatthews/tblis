#ifndef _TBLIS_TRACE_HPP_
#define _TBLIS_TRACE_HPP_

#include "tblis_marray.hpp"

namespace tblis
{

/*******************************************************************************
 *
 * Sum over (semi)diagonal elements of a tensor and sum onto a second
 *
 * The general form for a trace operation is ab...k*l*... -> ab... where k*
 * denotes the index k appearing one or more times, etc. and where the indices
 * kl... will be summed (traced) over. Indices may be transposed, and multiple
 * appearances of the traced indices kl... need not appear together. Either set
 * of indices may be empty, with the special case that when no indices are
 * traced over, the result is the same as transpose.
 *
 ******************************************************************************/

template <typename T>
int tensor_trace(T alpha, const_tensor_view<T> A, std::string idx_A,
                 T  beta,       tensor_view<T> B, std::string idx_B);

template <typename T>
int tensor_trace_ref(const std::vector<idx_type>& len_A,
                     const std::vector<idx_type>& len_AB,
                     T alpha, const T* A, const std::vector<stride_type>& stride_A_A,
                                          const std::vector<stride_type>& stride_A_AB,
                     T  beta,       T* B, const std::vector<stride_type>& stride_B_AB);

}

#endif
