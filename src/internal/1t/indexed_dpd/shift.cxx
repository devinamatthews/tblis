#include "shift.hpp"
#include "internal/1t/dpd/shift.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void shift(const communicator& comm, const config& cfg,
           T alpha, T beta, bool conj_A, const indexed_dpd_varray_view<T>& A,
           const dim_vector& idx_A_A)
{
    A.for_each_index(
    [&](const dpd_varray_view<T>& local_A)
    {
        shift(comm, cfg, alpha, beta, conj_A, local_A, idx_A_A);
    });
}

#define FOREACH_TYPE(T) \
template void shift(const communicator& comm, const config& cfg, \
                    T alpha, T beta, bool conj_A, const indexed_dpd_varray_view<T>& A, \
                    const dim_vector&);
#include "configs/foreach_type.h"

}
}
