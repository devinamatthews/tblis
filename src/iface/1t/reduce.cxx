#include "reduce.h"

#include "util/macros.h"
#include "util/tensor.hpp"
#include "internal/1t/dense/reduce.hpp"
#include "internal/1t/dpd/reduce.hpp"
#include "internal/1t/indexed/reduce.hpp"
#include "internal/1t/indexed_dpd/reduce.hpp"

namespace tblis
{

extern "C"
{

void tblis_tensor_reduce(const tblis_comm* comm, const tblis_config* cfg,
                         reduce_t op, const tblis_tensor* A, const label_type* idx_A_,
                         tblis_scalar* result, len_type* idx)
{
    TBLIS_ASSERT(A->type == result->type);

    unsigned ndim_A = A->ndim;
    len_vector len_A;
    stride_vector stride_A;
    label_vector idx_A;
    diagonal(ndim_A, A->len, A->stride, idx_A_, len_A, stride_A, idx_A);

    fold(len_A, idx_A, stride_A);

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        if (A->alpha<T>() < T(0))
        {
            if (op == REDUCE_MIN) op = REDUCE_MAX;
            else if (op == REDUCE_MAX) op = REDUCE_MIN;
        }

        parallelize_if(
        [&](const communicator& comm)
        {
            internal::reduce<T>(comm, get_config(cfg), op, len_A,
                                static_cast<const T*>(A->data), stride_A,
                                result->get<T>(), *idx);
        }, comm);

        if (A->conj)
        {
            result->get<T>() = conj(result->get<T>());
        }

        if (op == REDUCE_SUM)
        {
            result->get<T>() *= A->alpha<T>();
        }
        else if (op == REDUCE_SUM_ABS || op == REDUCE_NORM_2)
        {
            result->get<T>() *= std::abs(A->alpha<T>());
        }
    })
}

}

template <typename T>
void reduce(const communicator& comm, reduce_t op,
            dpd_varray_view<const T> A, const label_type* idx_A,
            T& result, len_type& idx)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim_A = A.dimension();

    for (unsigned i = 1;i < ndim_A;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    dim_vector idx_A_A = range(ndim_A);

    internal::reduce<T>(comm, get_default_config(), op,
                        A, idx_A_A, result, idx);
}

#define FOREACH_TYPE(T) \
template void reduce(const communicator& comm, reduce_t op, \
                     dpd_varray_view<const T> A, const label_type* idx_A, \
                     T& result, len_type& idx);
#include "configs/foreach_type.h"

template <typename T>
void reduce(const communicator& comm, reduce_t op,
            indexed_varray_view<const T> A, const label_type* idx_A,
            T& result, len_type& idx)
{
    unsigned ndim_A = A.dimension();

    for (unsigned i = 1;i < ndim_A;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    dim_vector idx_A_A = range(ndim_A);

    internal::reduce<T>(comm, get_default_config(), op,
                        A, idx_A_A, result, idx);
}

#define FOREACH_TYPE(T) \
template void reduce(const communicator& comm, reduce_t op, \
                     indexed_varray_view<const T> A, const label_type* idx_A, \
                     T& result, len_type& idx);
#include "configs/foreach_type.h"

template <typename T>
void reduce(const communicator& comm, reduce_t op,
            indexed_dpd_varray_view<const T> A, const label_type* idx_A,
            T& result, len_type& idx)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim_A = A.dimension();

    for (unsigned i = 1;i < ndim_A;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    dim_vector idx_A_A = range(ndim_A);

    internal::reduce<T>(comm, get_default_config(), op,
                        A, idx_A_A, result, idx);
}

#define FOREACH_TYPE(T) \
template void reduce(const communicator& comm, reduce_t op, \
                     indexed_dpd_varray_view<const T> A, const label_type* idx_A, \
                     T& result, len_type& idx);
#include "configs/foreach_type.h"

}
