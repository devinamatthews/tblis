#include "scale.hpp"
#include "internal/1t/dpd/scale.hpp"
#include "internal/1t/dpd/set.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void scale(const communicator& comm, const config& cfg,
           T alpha, bool conj_A, const indexed_dpd_varray_view<T>& A,
           const dim_vector& idx_A_A)
{
    auto local_A = A[0];
    auto diff = local_A.data() - A.data(0);

    for (len_type i = 0;i < A.num_indices();i++)
    {
        local_A.data(A.data(i) + diff);

        if (A.factor(i) == T(0))
        {
            set(comm, cfg, T(0), local_A, idx_A_A);
        }
        else
        {
            scale(comm, cfg, A.factor(i)*alpha, conj_A, local_A, idx_A_A);
        }
    }
}

#define FOREACH_TYPE(T) \
template void scale(const communicator& comm, const config& cfg, \
                    T alpha, bool conj_A, const indexed_dpd_varray_view<T>& A, \
                    const dim_vector&);
#include "configs/foreach_type.h"

}
}
