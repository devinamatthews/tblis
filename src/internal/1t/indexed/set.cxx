#include "set.hpp"
#include "internal/1t/dense/set.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void set(const communicator& comm, const config& cfg,
         T alpha, const indexed_varray_view<T>& A, const dim_vector& idx_A_A)
{
    A.for_each_index(
    [&](const varray_view<T>& local_A)
    {
        set(comm, cfg, local_A.lengths(), alpha, local_A.data(), local_A.strides());
    });
}

#define FOREACH_TYPE(T) \
template void set(const communicator& comm, const config& cfg, \
                  T alpha, const indexed_varray_view<T>& A, const dim_vector&);
#include "configs/foreach_type.h"

}
}
