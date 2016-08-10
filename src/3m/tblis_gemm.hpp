#ifndef _TBLIS_GEMM_HPP_
#define _TBLIS_GEMM_HPP_

#include "tblis_marray.hpp"
#include "tblis_thread.hpp"

namespace tblis
{

template <typename T>
void tblis_gemm(thread_communicator& comm,
                T alpha, const_matrix_view<T> A,
                         const_matrix_view<T> B,
                T  beta,       matrix_view<T> C);

template <typename T>
void tblis_gemm(T alpha, const_matrix_view<T> A,
                         const_matrix_view<T> B,
                T  beta,       matrix_view<T> C);

}

#endif
