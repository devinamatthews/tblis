#ifndef _TBLIS_ZEROV_HPP_
#define _TBLIS_ZEROV_HPP_

#include "tblis_marray.hpp"
#include "tblis_thread.hpp"

namespace tblis
{

template <typename T>
void tblis_zerov_ref(thread_communicator& comm, idx_type n, T* A, stride_type inc_A);

template <typename T> void tblis_zerov(idx_type n, T* A, stride_type inc_A);

template <typename T> void tblis_zerov(row_view<T> A);

}

#endif
