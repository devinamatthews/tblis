#include <tblis/internal/indexed_dpd.hpp>

namespace tblis
{

TBLIS_EXPORT
void tblis_tensor_scale(const tblis_comm* comm,
                        const tblis_config* cfg,
                              tblis_tensor* A,
                        const label_type* idx_A_)
{
    len_vector len_A(A->len, A->len+A->ndim);
    stride_vector stride_A(A->stride, A->stride+A->ndim);
    label_vector idx_A(idx_A_, idx_A_+A->ndim);
    internal::canonicalize(len_A, stride_A, idx_A);

    internal::fold(len_A, stride_A, idx_A);

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
           T alpha, dpd_marray_view<T> A, const label_string& idx_A)
{
    (void)idx_A;

    auto ndim_A = A.dimension();

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A.idx[i] != idx_A.idx[j]);

    dim_vector idx_A_A = range(ndim_A);

    if (alpha == T(0))
    {
        internal::set(type_tag<T>::value, comm, get_default_config(), alpha,
                      reinterpret_cast<dpd_marray_view<char>&>(A), idx_A_A);
    }
    else if (alpha != T(1))
    {
        internal::scale(type_tag<T>::value, comm, get_default_config(), alpha, false,
                        reinterpret_cast<dpd_marray_view<char>&>(A), idx_A_A);
    }
}

#define FOREACH_TYPE(T) \
template void scale(const communicator& comm, \
                    T alpha, dpd_marray_view<T> A, const label_string& idx_A);
#include <tblis/internal/foreach_type.h>

template <typename T>
void scale(const communicator& comm,
           T alpha, indexed_marray_view<T> A, const label_string& idx_A)
{
    (void)idx_A;

    auto ndim_A = A.dimension();

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A.idx[i] != idx_A.idx[j]);

    dim_vector idx_A_A = range(ndim_A);

    if (alpha == T(0))
    {
        internal::set(type_tag<T>::value, comm, get_default_config(), alpha,
                      reinterpret_cast<indexed_marray_view<char>&>(A), idx_A_A);
    }
    else if (alpha != T(1))
    {
        internal::scale(type_tag<T>::value, comm, get_default_config(), alpha, false,
                        reinterpret_cast<indexed_marray_view<char>&>(A), idx_A_A);
    }
}

#define FOREACH_TYPE(T) \
template void scale(const communicator& comm, \
                    T alpha, indexed_marray_view<T> A, const label_string& idx_A);
#include <tblis/internal/foreach_type.h>

template <typename T>
void scale(const communicator& comm,
           T alpha, indexed_dpd_marray_view<T> A, const label_string& idx_A)
{
    (void)idx_A;

    auto ndim_A = A.dimension();

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A.idx[i] != idx_A.idx[j]);

    dim_vector idx_A_A = range(ndim_A);

    if (alpha == T(0))
    {
        internal::set(type_tag<T>::value, comm, get_default_config(), alpha,
                      reinterpret_cast<indexed_dpd_marray_view<char>&>(A), idx_A_A);
    }
    else if (alpha != T(1))
    {
        internal::scale(type_tag<T>::value, comm, get_default_config(), alpha, false,
                        reinterpret_cast<indexed_dpd_marray_view<char>&>(A), idx_A_A);
    }
}

#define FOREACH_TYPE(T) \
template void scale(const communicator& comm, \
                    T alpha, indexed_dpd_marray_view<T> A, const label_string& idx_A);
#include <tblis/internal/foreach_type.h>

void scale(const communicator& comm,
           const scalar& alpha,
           const tensor& A_,
           const label_string& idx_A)
{
    tensor A(A_);
    A.scalar *= alpha.convert(A.type);
    tblis_tensor_scale(comm, nullptr, &A, idx_A.idx);
}

}
