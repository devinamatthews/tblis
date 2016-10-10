#ifndef _TBLIS_ZEROV_HPP_
#define _TBLIS_ZEROV_HPP_

#include "../../external/tci/src/tblis_thread.hpp"
#include "tblis_marray.hpp"

namespace tblis
{

template <typename T>
void tblis_zerov_ref(thread_communicator& comm, len_type n, T* A, stride_type inc_A);

template <typename T> void tblis_zerov(len_type n, T* A, stride_type inc_A);

template <typename T> void tblis_zerov(row_view<T> A);

}

#endif
