#include <tblis/internal/indexed_dpd.hpp>

namespace tblis
{

TBLIS_EXPORT
void tblis_tensor_reduce(const tblis_comm* comm,
                         const tblis_config* cfg,
                         reduce_t op,
                         const tblis_tensor* A,
                         const label_type* idx_A_,
                         tblis_scalar* result,
                         len_type* idx)
{
    TBLIS_ASSERT(A->type == result->type);

    len_vector len_A(A->len, A->len+A->ndim);
    stride_vector stride_A(A->stride, A->stride+A->ndim);
    label_vector idx_A(idx_A_, idx_A_+A->ndim);
    internal::canonicalize(len_A, stride_A, idx_A);

    internal::fold(len_A, stride_A, idx_A);

    if (A->scalar.is_negative())
    {
        if (op == REDUCE_MIN) op = REDUCE_MAX;
        else if (op == REDUCE_MAX) op = REDUCE_MIN;
    }

    parallelize_if(
    [&](const communicator& comm)
    {
        internal::reduce(A->type, comm, get_config(cfg), op, len_A,
                         reinterpret_cast<char*>(A->data), stride_A,
                         result->raw(), *idx);
    }, comm);

    if (A->conj) result->conj();

    if (op == REDUCE_SUM_ABS || op == REDUCE_MAX_ABS || op == REDUCE_MIN_ABS || op == REDUCE_NORM_2)
        *result *= abs(A->scalar);
    else
        *result *= A->scalar;
}

template <typename T>
void reduce(const communicator& comm, reduce_t op,
            dpd_varray_view<const T> A, const label_string& idx_A,
            T& result, len_type& idx)
{
    (void)idx_A;

    int ndim_A = A.dimension();

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A.idx[i] != idx_A.idx[j]);

    dim_vector idx_A_A = range(ndim_A);

    internal::reduce(type_tag<T>::value, comm, get_default_config(), op,
                     reinterpret_cast<dpd_varray_view<char>&>(A), idx_A_A,
                     reinterpret_cast<char*>(&result), idx);
}

#define FOREACH_TYPE(T) \
template void reduce(const communicator& comm, reduce_t op, \
                     dpd_varray_view<const T> A, const label_string& idx_A, \
                     T& result, len_type& idx);
#include <tblis/internal/foreach_type.h>

template <typename T>
void reduce(const communicator& comm, reduce_t op,
            indexed_varray_view<const T> A, const label_string& idx_A,
            T& result, len_type& idx)
{
    (void)idx_A;

    int ndim_A = A.dimension();

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A.idx[i] != idx_A.idx[j]);

    dim_vector idx_A_A = range(ndim_A);

    internal::reduce(type_tag<T>::value, comm, get_default_config(), op,
                     reinterpret_cast<indexed_varray_view<char>&>(A), idx_A_A,
                     reinterpret_cast<char*>(&result), idx);
}

#define FOREACH_TYPE(T) \
template void reduce(const communicator& comm, reduce_t op, \
                     indexed_varray_view<const T> A, const label_string& idx_A, \
                     T& result, len_type& idx);
#include <tblis/internal/foreach_type.h>

template <typename T>
void reduce(const communicator& comm, reduce_t op,
            indexed_dpd_varray_view<const T> A, const label_string& idx_A,
            T& result, len_type& idx)
{
    (void)idx_A;

    int ndim_A = A.dimension();

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A.idx[i] != idx_A.idx[j]);

    dim_vector idx_A_A = range(ndim_A);

    internal::reduce(type_tag<T>::value, comm, get_default_config(), op,
                     reinterpret_cast<indexed_dpd_varray_view<char>&>(A), idx_A_A,
                     reinterpret_cast<char*>(&result), idx);
}

#define FOREACH_TYPE(T) \
template void reduce(const communicator& comm, reduce_t op, \
                     indexed_dpd_varray_view<const T> A, const label_string& idx_A, \
                     T& result, len_type& idx);
#include <tblis/internal/foreach_type.h>

}
