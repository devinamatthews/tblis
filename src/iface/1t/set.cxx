#include "set.h"

#include "util/macros.h"
#include "util/tensor.hpp"
#include "internal/1t/set.hpp"
#include "internal/1t/dpd_set.hpp"

namespace tblis
{

extern "C"
{

void tblis_tensor_set(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_scalar* alpha, tblis_tensor* A, const label_type* idx_A_)
{
    TBLIS_ASSERT(alpha->type == A->type);

    unsigned ndim_A = A->ndim;
    std::vector<len_type> len_A;
    std::vector<stride_type> stride_A;
    std::vector<label_type> idx_A;
    diagonal(ndim_A, A->len, A->stride, idx_A_, len_A, stride_A, idx_A);

    fold(len_A, idx_A, stride_A);

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        parallelize_if(internal::set<T>, comm, get_config(cfg), len_A,
                       alpha->get<T>(), static_cast<T*>(A->data), stride_A);

        A->alpha<T>() = T(1);
        A->conj = false;
    })
}

}

template <typename T>
void set(const communicator& comm,
         T alpha, dpd_varray_view<T> A, const label_type* idx_A)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim_A = A.dimension();

    for (unsigned i = 1;i < ndim_A;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    std::vector<unsigned> idx_A_A = range(ndim_A);

    internal::dpd_set<T>(comm, get_default_config(), alpha, A, idx_A_A);
}

#define FOREACH_TYPE(T) \
template void set(const communicator& comm, \
                   T alpha, dpd_varray_view<T> A, const label_type* idx_A);
#include "configs/foreach_type.h"

}
