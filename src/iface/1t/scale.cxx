#include "scale.h"

#include "util/macros.h"
#include "util/tensor.hpp"
#include "internal/1t/dense/scale.hpp"
#include "internal/1t/dense/set.hpp"
#include "internal/1t/dpd/scale.hpp"
#include "internal/1t/dpd/set.hpp"
#include "internal/1t/indexed/scale.hpp"
#include "internal/1t/indexed/set.hpp"
#include "internal/1t/indexed_dpd/scale.hpp"
#include "internal/1t/indexed_dpd/set.hpp"

namespace tblis
{

TBLIS_EXPORT
void tblis_tensor_scale(const tblis_comm* comm,
                        const tblis_config* cfg,
                              tblis_tensor* A,
                        const label_type* idx_A_)
{
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
        if (A->scalar.is_zero())
        {
            internal::set(A->type, comm, get_config(cfg), len_A,
                          A->scalar, reinterpret_cast<char*>(A->data), stride_A);
        }
        else if (!A->scalar.is_one() || (A->scalar.is_complex() && A->conj))
        {
            internal::scale(A->type, comm, get_config(cfg), len_A,
                            A->scalar, A->conj,
                            reinterpret_cast<char*>(A->data), stride_A);
        }
    }, comm);

    A->scalar = 1;
    A->conj = false;
}

template <typename T>
void scale(const communicator& comm,
           T alpha, dpd_varray_view<T> A, const label_vector& idx_A)
{
    (void)idx_A;

    auto ndim_A = A.dimension();

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    dim_vector idx_A_A = range(ndim_A);

    if (alpha == T(0))
    {
        internal::set(type_tag<T>::value, comm, get_default_config(), alpha,
                      reinterpret_cast<dpd_varray_view<char>&>(A), idx_A_A);
    }
    else if (alpha != T(1))
    {
        internal::scale(type_tag<T>::value, comm, get_default_config(), alpha, false,
                        reinterpret_cast<dpd_varray_view<char>&>(A), idx_A_A);
    }
}

#define FOREACH_TYPE(T) \
template void scale(const communicator& comm, \
                    T alpha, dpd_varray_view<T> A, const label_vector& idx_A);
#include "configs/foreach_type.h"

template <typename T>
void scale(const communicator& comm,
           T alpha, indexed_varray_view<T> A, const label_vector& idx_A)
{
    (void)idx_A;

    auto ndim_A = A.dimension();

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    dim_vector idx_A_A = range(ndim_A);

    if (alpha == T(0))
    {
        internal::set(type_tag<T>::value, comm, get_default_config(), alpha,
                      reinterpret_cast<indexed_varray_view<char>&>(A), idx_A_A);
    }
    else if (alpha != T(1))
    {
        internal::scale(type_tag<T>::value, comm, get_default_config(), alpha, false,
                        reinterpret_cast<indexed_varray_view<char>&>(A), idx_A_A);
    }
}

#define FOREACH_TYPE(T) \
template void scale(const communicator& comm, \
                    T alpha, indexed_varray_view<T> A, const label_vector& idx_A);
#include "configs/foreach_type.h"

template <typename T>
void scale(const communicator& comm,
           T alpha, indexed_dpd_varray_view<T> A, const label_vector& idx_A)
{
    (void)idx_A;

    auto ndim_A = A.dimension();

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    dim_vector idx_A_A = range(ndim_A);

    if (alpha == T(0))
    {
        internal::set(type_tag<T>::value, comm, get_default_config(), alpha,
                      reinterpret_cast<indexed_dpd_varray_view<char>&>(A), idx_A_A);
    }
    else if (alpha != T(1))
    {
        internal::scale(type_tag<T>::value, comm, get_default_config(), alpha, false,
                        reinterpret_cast<indexed_dpd_varray_view<char>&>(A), idx_A_A);
    }
}

#define FOREACH_TYPE(T) \
template void scale(const communicator& comm, \
                    T alpha, indexed_dpd_varray_view<T> A, const label_vector& idx_A);
#include "configs/foreach_type.h"

}
