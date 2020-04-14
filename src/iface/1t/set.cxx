#include "set.h"

#include "util/macros.h"
#include "util/tensor.hpp"
#include "internal/1t/dense/set.hpp"
#include "internal/1t/dpd/set.hpp"
#include "internal/1t/indexed/set.hpp"
#include "internal/1t/indexed_dpd/set.hpp"

namespace tblis
{

TBLIS_EXPORT
void tblis_tensor_set(const tblis_comm* comm,
                      const tblis_config* cfg,
                      const tblis_scalar* alpha,
                            tblis_tensor* A,
                      const label_type* idx_A_)
{
    TBLIS_ASSERT(alpha->type == A->type);

    auto ndim_A = A->ndim;
    len_vector len_A;
    stride_vector stride_A;
    label_vector idx_A;
    diagonal(ndim_A, A->len, A->stride, idx_A_, len_A, stride_A, idx_A);

    if (idx_A.empty())
    {
        len_A.push_back(1);
        stride_A.push_back(0);
        idx_A.push_back(0);
    }

    fold(len_A, idx_A, stride_A);

    parallelize_if(
    [&](const communicator& comm)
    {
        internal::set(A->type, comm, get_config(cfg), len_A,
                      *alpha, reinterpret_cast<char*>(A->data), stride_A);
    }, comm);

    A->scalar = 1;
    A->conj = false;
}

template <typename T>
void set(const communicator& comm,
         T alpha, dpd_varray_view<T> A, const label_vector& idx_A)
{
    (void)idx_A;

    auto ndim_A = A.dimension();

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    dim_vector idx_A_A = range(ndim_A);

    internal::set(type_tag<T>::value, comm, get_default_config(), alpha,
                  reinterpret_cast<dpd_varray_view<char>&>(A), idx_A_A);
}

#define FOREACH_TYPE(T) \
template void set(const communicator& comm, \
                   T alpha, dpd_varray_view<T> A, const label_vector& idx_A);
#include "configs/foreach_type.h"

template <typename T>
void set(const communicator& comm,
         T alpha, indexed_varray_view<T> A, const label_vector& idx_A)
{
    (void)idx_A;

    auto ndim_A = A.dimension();

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    dim_vector idx_A_A = range(ndim_A);

    internal::set(type_tag<T>::value, comm, get_default_config(), alpha,
                  reinterpret_cast<indexed_varray_view<char>&>(A), idx_A_A);
}

#define FOREACH_TYPE(T) \
template void set(const communicator& comm, \
                   T alpha, indexed_varray_view<T> A, const label_vector& idx_A);
#include "configs/foreach_type.h"

template <typename T>
void set(const communicator& comm,
         T alpha, indexed_dpd_varray_view<T> A, const label_vector& idx_A)
{
    (void)idx_A;

    auto ndim_A = A.dimension();

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    dim_vector idx_A_A = range(ndim_A);

    internal::set(type_tag<T>::value, comm, get_default_config(), alpha,
                  reinterpret_cast<indexed_dpd_varray_view<char>&>(A), idx_A_A);
}

#define FOREACH_TYPE(T) \
template void set(const communicator& comm, \
                   T alpha, indexed_dpd_varray_view<T> A, const label_vector& idx_A);
#include "configs/foreach_type.h"

}
