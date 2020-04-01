#ifndef _TBLIS_IFACE_1T_REDUCE_H_
#define _TBLIS_IFACE_1T_REDUCE_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

#ifdef __cplusplus
namespace tblis
{
#endif

TBLIS_EXPORT

void tblis_tensor_reduce(const tblis_comm* comm,
                         const tblis_config* cfg,
                         reduce_t op,
                         const tblis_tensor* A,
                         const label_type* idx_A,
                         tblis_scalar* result,
                         len_type* idx);

#if defined(__cplusplus)

template <typename T=scalar>
struct reduce_result
{
    T value;
    len_type idx;

    template <typename T_=T, typename =
        std::enable_if_t<!std::is_same<T_,scalar>::value>>
    reduce_result()
    : value(), idx() {}

    reduce_result(const T& value, len_type idx)
    : value(value), idx(idx) {}

    operator const T&() const { return value; }
};

inline
void reduce(const communicator& comm,
            reduce_t op,
            const tensor& A,
            const label_vector& idx_A,
            scalar& result,
            len_type& idx)
{
    tblis_tensor_reduce(comm, nullptr, op, &A, idx_A.data(), &result, &idx);
}

template <typename T>
void reduce(const communicator& comm,
            reduce_t op,
            const tensor& A,
            const label_vector& idx_A,
            T& result,
            len_type& idx)
{
    scalar result_(0.0, A.type);
    reduce(comm, op, A, idx_A, result_, idx);
    result = result_.get<T>();
}

inline
reduce_result<scalar> reduce(const communicator& comm,
                             reduce_t op,
                             const tensor& A,
                             const label_vector& idx_A)
{
    reduce_result<scalar> result({0, A.type}, 0);
    reduce(comm, op, A, idx_A, result.value, result.idx);
    return result;
}

template <typename T>
reduce_result<T> reduce(const communicator& comm,
                        reduce_t op,
                        const tensor& A,
                        const label_vector& idx_A)
{
    reduce_result<T> result;
    reduce(comm, op, A, idx_A, result.value, result.idx);
    return result;
}

inline
void reduce(const communicator& comm,
            reduce_t op,
            const tensor& A,
            scalar& result,
            len_type& idx)
{
    reduce(comm, op, A, tblis::idx(A), result, idx);
}

template <typename T>
void reduce(const communicator& comm,
            reduce_t op,
            const tensor& A,
            T& result,
            len_type& idx)
{
    scalar result_(0.0, A.type);
    reduce(comm, op, A, result_, idx);
    result = result_.get<T>();
}

inline
reduce_result<scalar> reduce(const communicator& comm,
                             reduce_t op,
                             const tensor& A)
{
    reduce_result<scalar> result({0, A.type}, 0);
    reduce(comm, op, A, result.value, result.idx);
    return result;
}

template <typename T>
reduce_result<T> reduce(const communicator& comm,
                        reduce_t op,
                        const tensor& A)
{
    reduce_result<T> result;
    reduce(comm, op, A, result.value, result.idx);
    return result;
}

inline
void reduce(reduce_t op,
            const tensor& A,
            const label_vector& idx_A,
            scalar& result,
            len_type& idx)
{
    reduce(*(communicator*)nullptr, op, A, idx_A, result, idx);
}

template <typename T>
void reduce(reduce_t op,
            const tensor& A,
            const label_vector& idx_A,
            T& result,
            len_type& idx)
{
    scalar result_(0.0, A.type);
    reduce(op, A, idx_A, result_, idx);
    result = result_.get<T>();
}

inline
reduce_result<scalar> reduce(reduce_t op,
                             const tensor& A,
                             const label_vector& idx_A)
{
    reduce_result<scalar> result({0, A.type}, 0);
    reduce(op, A, idx_A, result.value, result.idx);
    return result;
}

template <typename T>
reduce_result<T> reduce(reduce_t op,
                        const tensor& A,
                        const label_vector& idx_A)
{
    reduce_result<T> result;
    reduce(op, A, idx_A, result.value, result.idx);
    return result;
}

inline
void reduce(reduce_t op,
            const tensor& A,
            scalar& result,
            len_type& idx)
{
    reduce(op, A, tblis::idx(A), result, idx);
}

template <typename T>
void reduce(reduce_t op,
            const tensor& A,
            T& result,
            len_type& idx)
{
    scalar result_(0.0, A.type);
    reduce(op, A, result_, idx);
    result = result_.get<T>();
}

inline
reduce_result<scalar> reduce(reduce_t op, const tensor& A)
{
    reduce_result<scalar> result({0, A.type}, 0);
    reduce(op, A, result.value, result.idx);
    return result;
}

template <typename T>
reduce_result<T> reduce(reduce_t op, const tensor& A)
{
    reduce_result<T> result;
    reduce(op, A, result.value, result.idx);
    return result;
}

#if !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void reduce(const communicator& comm, reduce_t op, dpd_varray_view<const T> A,
            const label_vector& idx_A, T& result, len_type& idx);

template <typename T>
void reduce(reduce_t op, dpd_varray_view<const T> A, const label_vector& idx_A,
            T& result, len_type& idx)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            reduce(comm, op, A, idx_A, result, idx);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
reduce_result<T> reduce(reduce_t op, dpd_varray_view<const T> A, const label_vector& idx_A)
{
    reduce_result<T> result;
    reduce(op, A, idx_A, result.value, result.idx);
    return result;
}

template <typename T>
reduce_result<T> reduce(const communicator& comm, reduce_t op,
                             dpd_varray_view<const T> A, const label_vector& idx_A)
{
    reduce_result<T> result;
    reduce(comm, op, A, idx_A, result.value, result.idx);
    return result;
}

template <typename T>
void reduce(const communicator& comm, reduce_t op, indexed_varray_view<const T> A,
            const label_vector& idx_A, T& result, len_type& idx);

template <typename T>
void reduce(reduce_t op, indexed_varray_view<const T> A, const label_vector& idx_A,
            T& result, len_type& idx)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            reduce(comm, op, A, idx_A, result, idx);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
reduce_result<T> reduce(reduce_t op, indexed_varray_view<const T> A, const label_vector& idx_A)
{
    reduce_result<T> result;
    reduce(op, A, idx_A, result.value, result.idx);
    return result;
}

template <typename T>
reduce_result<T> reduce(const communicator& comm, reduce_t op,
                             indexed_varray_view<const T> A, const label_vector& idx_A)
{
    reduce_result<T> result;
    reduce(comm, op, A, idx_A, result.value, result.idx);
    return result;
}

template <typename T>
void reduce(const communicator& comm, reduce_t op, indexed_dpd_varray_view<const T> A,
            const label_vector& idx_A, T& result, len_type& idx);

template <typename T>
void reduce(reduce_t op, indexed_dpd_varray_view<const T> A, const label_vector& idx_A,
            T& result, len_type& idx)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            reduce(comm, op, A, idx_A, result, idx);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
reduce_result<T> reduce(reduce_t op, indexed_dpd_varray_view<const T> A, const label_vector& idx_A)
{
    reduce_result<T> result;
    reduce(op, A, idx_A, result.value, result.idx);
    return result;
}

template <typename T>
reduce_result<T> reduce(const communicator& comm, reduce_t op,
                             indexed_dpd_varray_view<const T> A, const label_vector& idx_A)
{
    reduce_result<T> result;
    reduce(comm, op, A, idx_A, result.value, result.idx);
    return result;
}

#endif

}

#endif

#pragma GCC diagnostic pop

#endif
