#include <tblis/internal/indexed_dpd.hpp>

namespace tblis
{

TBLIS_EXPORT
void tblis_tensor_mult(const tblis_comm* comm,
                       const tblis_config* cfg,
                       const tblis_tensor* A,
                       const label_type* idx_A_,
                       const tblis_tensor* B,
                       const label_type* idx_B_,
                             tblis_tensor* C,
                       const label_type* idx_C_)
{
    TBLIS_ASSERT(A->type == B->type);
    TBLIS_ASSERT(A->type == C->type);

    len_vector len_A(A->len, A->len+A->ndim);
    stride_vector stride_A(A->stride, A->stride+A->ndim);
    label_vector idx_A(idx_A_, idx_A_+A->ndim);
    internal::canonicalize(len_A, stride_A, idx_A);

    len_vector len_B(B->len, B->len+B->ndim);
    stride_vector stride_B(B->stride, B->stride+B->ndim);
    label_vector idx_B(idx_B_, idx_B_+B->ndim);
    internal::canonicalize(len_B, stride_B, idx_B);

    len_vector len_C(C->len, C->len+C->ndim);
    stride_vector stride_C(C->stride, C->stride+C->ndim);
    label_vector idx_C(idx_C_, idx_C_+C->ndim);
    internal::canonicalize(len_C, stride_C, idx_C);

    auto idx_ABC = stl_ext::intersection(idx_A, idx_B, idx_C);
    auto len_ABC = stl_ext::select_from(len_A, idx_A, idx_ABC);
    TBLIS_ASSERT(len_ABC == stl_ext::select_from(len_B, idx_B, idx_ABC));
    TBLIS_ASSERT(len_ABC == stl_ext::select_from(len_C, idx_C, idx_ABC));
    auto stride_A_ABC = stl_ext::select_from(stride_A, idx_A, idx_ABC);
    auto stride_B_ABC = stl_ext::select_from(stride_B, idx_B, idx_ABC);
    auto stride_C_ABC = stl_ext::select_from(stride_C, idx_C, idx_ABC);

    auto idx_AB = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_B), idx_ABC);
    auto len_AB = stl_ext::select_from(len_A, idx_A, idx_AB);
    TBLIS_ASSERT(len_AB == stl_ext::select_from(len_B, idx_B, idx_AB));
    auto stride_A_AB = stl_ext::select_from(stride_A, idx_A, idx_AB);
    auto stride_B_AB = stl_ext::select_from(stride_B, idx_B, idx_AB);

    auto idx_AC = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_C), idx_ABC);
    auto len_AC = stl_ext::select_from(len_A, idx_A, idx_AC);
    TBLIS_ASSERT(len_AC == stl_ext::select_from(len_C, idx_C, idx_AC));
    auto stride_A_AC = stl_ext::select_from(stride_A, idx_A, idx_AC);
    auto stride_C_AC = stl_ext::select_from(stride_C, idx_C, idx_AC);

    auto idx_BC = stl_ext::exclusion(stl_ext::intersection(idx_B, idx_C), idx_ABC);
    auto len_BC = stl_ext::select_from(len_B, idx_B, idx_BC);
    TBLIS_ASSERT(len_BC == stl_ext::select_from(len_C, idx_C, idx_BC));
    auto stride_B_BC = stl_ext::select_from(stride_B, idx_B, idx_BC);
    auto stride_C_BC = stl_ext::select_from(stride_C, idx_C, idx_BC);

    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB, idx_AC, idx_ABC);
    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB, idx_BC, idx_ABC);
    auto idx_C_only = stl_ext::exclusion(idx_C, idx_AC, idx_BC, idx_ABC);

    TBLIS_ASSERT(idx_A_only.empty());
    TBLIS_ASSERT(idx_B_only.empty());
    TBLIS_ASSERT(idx_C_only.empty());

    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_AC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AC, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AC, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_BC, idx_ABC).empty());

    internal::fold(len_ABC, stride_C_ABC, stride_A_ABC, stride_B_ABC, idx_ABC);
    internal::fold(len_AB, stride_A_AB, stride_B_AB, idx_AB);
    internal::fold(len_AC, stride_C_AC, stride_A_AC, idx_AC);
    internal::fold(len_BC, stride_C_BC, stride_B_BC, idx_BC);

    len_vector nolen;
    stride_vector nostride;

    auto alpha = A->scalar*B->scalar;
    auto beta = C->scalar;

    auto data_A = reinterpret_cast<char*>(A->data);
    auto data_B = reinterpret_cast<char*>(B->data);
    auto data_C = reinterpret_cast<char*>(C->data);

    parallelize_if(
    [&](const communicator& comm)
    {
        if (alpha.is_zero())
        {
            len_ABC.insert(len_ABC.end(), len_AC.begin(), len_AC.end());
            len_ABC.insert(len_ABC.end(), len_BC.begin(), len_BC.end());
            stride_C_ABC.insert(stride_C_ABC.end(), stride_C_AC.begin(), stride_C_AC.end());
            stride_C_ABC.insert(stride_C_ABC.end(), stride_C_BC.begin(), stride_C_BC.end());

            if (beta.is_zero())
            {
                internal::set(A->type, comm, get_config(cfg),
                              len_ABC, beta, data_C,
                              stride_C_ABC);
            }
            else if (!beta.is_one() || (beta.is_complex() && C->conj))
            {
                internal::scale(A->type, comm, get_config(cfg),
                                len_ABC, beta, C->conj, data_C,
                                stride_C_ABC);
            }
        }
        else
        {
            internal::mult(A->type, comm, get_config(cfg),
                           len_AB, len_AC, len_BC, len_ABC,
                           alpha, A->conj, data_A,
                           stride_A_AB, stride_A_AC, stride_A_ABC,
                                  B->conj, data_B,
                           stride_B_AB, stride_B_BC, stride_B_ABC,
                            beta, C->conj, data_C,
                           stride_C_AC, stride_C_BC, stride_C_ABC);
        }
    }, comm);

    C->scalar = 1;
    C->conj = false;
}

template <typename T>
void mult(const communicator& comm,
          T alpha, const dpd_varray_view<const T>& A, const label_string& idx_A_,
                   const dpd_varray_view<const T>& B, const label_string& idx_B_,
          T  beta, const dpd_varray_view<      T>& C, const label_string& idx_C_)
{
    auto nirrep = A.num_irreps();
    TBLIS_ASSERT(B.num_irreps() == nirrep);
    TBLIS_ASSERT(C.num_irreps() == nirrep);

    int ndim_A = A.dimension();
    int ndim_B = B.dimension();
    int ndim_C = C.dimension();

    label_vector idx_A(idx_A_.idx, idx_A_.idx+ndim_A);
    label_vector idx_B(idx_B_.idx, idx_B_.idx+ndim_B);
    label_vector idx_C(idx_C_.idx, idx_C_.idx+ndim_C);

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    for (auto i : range(1,ndim_B))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_B[i] != idx_B[j]);

    for (auto i : range(1,ndim_C))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_C[i] != idx_C[j]);

    auto idx_ABC = stl_ext::intersection(idx_A, idx_B, idx_C);
    auto idx_AB = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_B), idx_ABC);
    auto idx_AC = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_C), idx_ABC);
    auto idx_BC = stl_ext::exclusion(stl_ext::intersection(idx_B, idx_C), idx_ABC);
    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB, idx_AC, idx_ABC);
    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB, idx_BC, idx_ABC);
    auto idx_C_only = stl_ext::exclusion(idx_C, idx_AC, idx_BC, idx_ABC);

    TBLIS_ASSERT(idx_A_only.empty());
    TBLIS_ASSERT(idx_B_only.empty());
    TBLIS_ASSERT(idx_C_only.empty());

    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_AC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AC, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AC, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_BC, idx_ABC).empty());

    dim_vector range_A = range(ndim_A);
    dim_vector range_B = range(ndim_B);
    dim_vector range_C = range(ndim_C);

    auto idx_A_ABC = stl_ext::select_from(range_A, idx_A, idx_ABC);
    auto idx_B_ABC = stl_ext::select_from(range_B, idx_B, idx_ABC);
    auto idx_C_ABC = stl_ext::select_from(range_C, idx_C, idx_ABC);
    auto idx_A_AB = stl_ext::select_from(range_A, idx_A, idx_AB);
    auto idx_B_AB = stl_ext::select_from(range_B, idx_B, idx_AB);
    auto idx_A_AC = stl_ext::select_from(range_A, idx_A, idx_AC);
    auto idx_C_AC = stl_ext::select_from(range_C, idx_C, idx_AC);
    auto idx_B_BC = stl_ext::select_from(range_B, idx_B, idx_BC);
    auto idx_C_BC = stl_ext::select_from(range_C, idx_C, idx_BC);

    for (auto i : range(idx_ABC.size()))
    for (auto irrep : range(nirrep))
    {
        TBLIS_ASSERT(A.length(idx_A_ABC[i], irrep) ==
                     B.length(idx_B_ABC[i], irrep));
        TBLIS_ASSERT(A.length(idx_A_ABC[i], irrep) ==
                     C.length(idx_C_ABC[i], irrep));
    }

    for (auto i : range(idx_AB.size()))
    for (auto irrep : range(nirrep))
    {
        TBLIS_ASSERT(A.length(idx_A_AB[i], irrep) ==
                     B.length(idx_B_AB[i], irrep));
    }

    for (auto i : range(idx_AC.size()))
    for (auto irrep : range(nirrep))
    {
        TBLIS_ASSERT(A.length(idx_A_AC[i], irrep) ==
                     C.length(idx_C_AC[i], irrep));
    }

    for (auto i : range(idx_BC.size()))
    for (auto irrep : range(nirrep))
    {
        TBLIS_ASSERT(B.length(idx_B_BC[i], irrep) ==
                     C.length(idx_C_BC[i], irrep));
    }

    if (alpha == T(0) || (idx_ABC.empty() && ((A.irrep()^B.irrep()) != C.irrep())))
    {
        if (beta == T(0))
        {
            internal::set(type_tag<T>::value, comm, get_default_config(),
                          beta, reinterpret_cast<const dpd_varray_view<char>&>(C), range_C);
        }
        else if (beta != T(1))
        {
            internal::scale(type_tag<T>::value, comm, get_default_config(),
                            beta, false, reinterpret_cast<const dpd_varray_view<char>&>(C), range_C);
        }
    }
    else
    {
        internal::mult(type_tag<T>::value, comm, get_default_config(),
                       alpha, false, reinterpret_cast<const dpd_varray_view<char>&>(A), idx_A_AB, idx_A_AC, idx_A_ABC,
                              false, reinterpret_cast<const dpd_varray_view<char>&>(B), idx_B_AB, idx_B_BC, idx_B_ABC,
                        beta, false, reinterpret_cast<const dpd_varray_view<char>&>(C), idx_C_AC, idx_C_BC, idx_C_ABC);
    }
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, \
                   T alpha, const dpd_varray_view<const T>& A, const label_string& idx_A, \
                            const dpd_varray_view<const T>& B, const label_string& idx_B, \
                   T  beta, const dpd_varray_view<      T>& C, const label_string& idx_C);
#include <tblis/internal/foreach_type.h>

template <typename T>
void mult(const communicator& comm,
          T alpha, const indexed_varray_view<const T>& A, const label_string& idx_A_,
                   const indexed_varray_view<const T>& B, const label_string& idx_B_,
          T  beta, const indexed_varray_view<      T>& C, const label_string& idx_C_)
{
    int ndim_A = A.dimension();
    int ndim_B = B.dimension();
    int ndim_C = C.dimension();

    label_vector idx_A(idx_A_.idx, idx_A_.idx+ndim_A);
    label_vector idx_B(idx_B_.idx, idx_B_.idx+ndim_B);
    label_vector idx_C(idx_C_.idx, idx_C_.idx+ndim_C);

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    for (auto i : range(1,ndim_B))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_B[i] != idx_B[j]);

    for (auto i : range(1,ndim_C))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_C[i] != idx_C[j]);

    auto idx_ABC = stl_ext::intersection(idx_A, idx_B, idx_C);
    auto idx_AB = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_B), idx_ABC);
    auto idx_AC = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_C), idx_ABC);
    auto idx_BC = stl_ext::exclusion(stl_ext::intersection(idx_B, idx_C), idx_ABC);
    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB, idx_AC, idx_ABC);
    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB, idx_BC, idx_ABC);
    auto idx_C_only = stl_ext::exclusion(idx_C, idx_AC, idx_BC, idx_ABC);

    TBLIS_ASSERT(idx_A_only.empty());
    TBLIS_ASSERT(idx_B_only.empty());
    TBLIS_ASSERT(idx_C_only.empty());

    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_AC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AC, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AC, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_BC, idx_ABC).empty());

    dim_vector range_A = range(ndim_A);
    dim_vector range_B = range(ndim_B);
    dim_vector range_C = range(ndim_C);

    auto idx_A_ABC = stl_ext::select_from(range_A, idx_A, idx_ABC);
    auto idx_B_ABC = stl_ext::select_from(range_B, idx_B, idx_ABC);
    auto idx_C_ABC = stl_ext::select_from(range_C, idx_C, idx_ABC);
    auto idx_A_AB = stl_ext::select_from(range_A, idx_A, idx_AB);
    auto idx_B_AB = stl_ext::select_from(range_B, idx_B, idx_AB);
    auto idx_A_AC = stl_ext::select_from(range_A, idx_A, idx_AC);
    auto idx_C_AC = stl_ext::select_from(range_C, idx_C, idx_AC);
    auto idx_B_BC = stl_ext::select_from(range_B, idx_B, idx_BC);
    auto idx_C_BC = stl_ext::select_from(range_C, idx_C, idx_BC);

    for (auto i : range(idx_ABC.size()))
    {
        TBLIS_ASSERT(A.length(idx_A_ABC[i]) ==
                     B.length(idx_B_ABC[i]));
        TBLIS_ASSERT(A.length(idx_A_ABC[i]) ==
                     C.length(idx_C_ABC[i]));
    }

    for (auto i : range(idx_AB.size()))
    {
        TBLIS_ASSERT(A.length(idx_A_AB[i]) ==
                     B.length(idx_B_AB[i]));
    }

    for (auto i : range(idx_AC.size()))
    {
        TBLIS_ASSERT(A.length(idx_A_AC[i]) ==
                     C.length(idx_C_AC[i]));
    }

    for (auto i : range(idx_BC.size()))
    {
        TBLIS_ASSERT(B.length(idx_B_BC[i]) ==
                     C.length(idx_C_BC[i]));
    }

    if (alpha == T(0))
    {
        if (beta == T(0))
        {
            internal::set(type_tag<T>::value, comm, get_default_config(),
                          beta, reinterpret_cast<const indexed_varray_view<char>&>(C), range_C);
        }
        else if (beta != T(1))
        {
            internal::scale(type_tag<T>::value, comm, get_default_config(),
                            beta, false, reinterpret_cast<const indexed_varray_view<char>&>(C), range_C);
        }
    }
    else
    {
        internal::mult(type_tag<T>::value, comm, get_default_config(),
                       alpha, false, reinterpret_cast<const indexed_varray_view<char>&>(A), idx_A_AB, idx_A_AC, idx_A_ABC,
                              false, reinterpret_cast<const indexed_varray_view<char>&>(B), idx_B_AB, idx_B_BC, idx_B_ABC,
                        beta, false, reinterpret_cast<const indexed_varray_view<char>&>(C), idx_C_AC, idx_C_BC, idx_C_ABC);
    }
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, \
                   T alpha, const indexed_varray_view<const T>& A, const label_string& idx_A, \
                            const indexed_varray_view<const T>& B, const label_string& idx_B, \
                   T  beta, const indexed_varray_view<      T>& C, const label_string& idx_C);
#include <tblis/internal/foreach_type.h>

template <typename T>
void mult(const communicator& comm,
          T alpha, const indexed_dpd_varray_view<const T>& A, const label_string& idx_A_,
                   const indexed_dpd_varray_view<const T>& B, const label_string& idx_B_,
          T  beta, const indexed_dpd_varray_view<      T>& C, const label_string& idx_C_)
{
    auto nirrep = A.num_irreps();
    TBLIS_ASSERT(B.num_irreps() == nirrep);
    TBLIS_ASSERT(C.num_irreps() == nirrep);

    int ndim_A = A.dimension();
    int ndim_B = B.dimension();
    int ndim_C = C.dimension();

    label_vector idx_A(idx_A_.idx, idx_A_.idx+ndim_A);
    label_vector idx_B(idx_B_.idx, idx_B_.idx+ndim_B);
    label_vector idx_C(idx_C_.idx, idx_C_.idx+ndim_C);

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    for (auto i : range(1,ndim_B))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_B[i] != idx_B[j]);

    for (auto i : range(1,ndim_C))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_C[i] != idx_C[j]);

    auto idx_ABC = stl_ext::intersection(idx_A, idx_B, idx_C);
    auto idx_AB = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_B), idx_ABC);
    auto idx_AC = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_C), idx_ABC);
    auto idx_BC = stl_ext::exclusion(stl_ext::intersection(idx_B, idx_C), idx_ABC);
    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB, idx_AC, idx_ABC);
    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB, idx_BC, idx_ABC);
    auto idx_C_only = stl_ext::exclusion(idx_C, idx_AC, idx_BC, idx_ABC);

    TBLIS_ASSERT(idx_A_only.empty());
    TBLIS_ASSERT(idx_B_only.empty());
    TBLIS_ASSERT(idx_C_only.empty());

    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_AC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AC, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AC, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_BC, idx_ABC).empty());

    dim_vector range_A = range(ndim_A);
    dim_vector range_B = range(ndim_B);
    dim_vector range_C = range(ndim_C);

    auto idx_A_ABC = stl_ext::select_from(range_A, idx_A, idx_ABC);
    auto idx_B_ABC = stl_ext::select_from(range_B, idx_B, idx_ABC);
    auto idx_C_ABC = stl_ext::select_from(range_C, idx_C, idx_ABC);
    auto idx_A_AB = stl_ext::select_from(range_A, idx_A, idx_AB);
    auto idx_B_AB = stl_ext::select_from(range_B, idx_B, idx_AB);
    auto idx_A_AC = stl_ext::select_from(range_A, idx_A, idx_AC);
    auto idx_C_AC = stl_ext::select_from(range_C, idx_C, idx_AC);
    auto idx_B_BC = stl_ext::select_from(range_B, idx_B, idx_BC);
    auto idx_C_BC = stl_ext::select_from(range_C, idx_C, idx_BC);

    for (auto i : range(idx_ABC.size()))
    for (auto irrep : range(nirrep))
    {
        TBLIS_ASSERT(A.length(idx_A_ABC[i], irrep) ==
                     B.length(idx_B_ABC[i], irrep));
        TBLIS_ASSERT(A.length(idx_A_ABC[i], irrep) ==
                     C.length(idx_C_ABC[i], irrep));
    }

    for (auto i : range(idx_AB.size()))
    for (auto irrep : range(nirrep))
    {
        TBLIS_ASSERT(A.length(idx_A_AB[i], irrep) ==
                     B.length(idx_B_AB[i], irrep));
    }

    for (auto i : range(idx_AC.size()))
    for (auto irrep : range(nirrep))
    {
        TBLIS_ASSERT(A.length(idx_A_AC[i], irrep) ==
                     C.length(idx_C_AC[i], irrep));
    }

    for (auto i : range(idx_BC.size()))
    for (auto irrep : range(nirrep))
    {
        TBLIS_ASSERT(B.length(idx_B_BC[i], irrep) ==
                     C.length(idx_C_BC[i], irrep));
    }

    if (alpha == T(0) || (idx_ABC.empty() && ((A.irrep()^B.irrep()) != C.irrep())))
    {
        if (beta == T(0))
        {
            internal::set(type_tag<T>::value, comm, get_default_config(),
                          beta, reinterpret_cast<const indexed_dpd_varray_view<char>&>(C), range_C);
        }
        else if (beta != T(1))
        {
            internal::scale(type_tag<T>::value, comm, get_default_config(),
                            beta, false, reinterpret_cast<const indexed_dpd_varray_view<char>&>(C), range_C);
        }
    }
    else
    {
        internal::mult(type_tag<T>::value, comm, get_default_config(),
                       alpha, false, reinterpret_cast<const indexed_dpd_varray_view<char>&>(A), idx_A_AB, idx_A_AC, idx_A_ABC,
                              false, reinterpret_cast<const indexed_dpd_varray_view<char>&>(B), idx_B_AB, idx_B_BC, idx_B_ABC,
                        beta, false, reinterpret_cast<const indexed_dpd_varray_view<char>&>(C), idx_C_AC, idx_C_BC, idx_C_ABC);
    }
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, \
                   T alpha, const indexed_dpd_varray_view<const T>& A, const label_string& idx_A, \
                            const indexed_dpd_varray_view<const T>& B, const label_string& idx_B, \
                   T  beta, const indexed_dpd_varray_view<      T>& C, const label_string& idx_C);
#include <tblis/internal/foreach_type.h>

void mult(const communicator& comm,
          const scalar& alpha,
          const const_tensor& A_,
          const label_string& idx_A,
          const const_tensor& B,
          const label_string& idx_B,
          const scalar& beta,
          const tensor& C_,
          const label_string& idx_C)
{
    tensor A(A_.tensor_);
    A.scalar *= alpha;

    tensor C(C_);
    C.scalar *= beta;

    tblis_tensor_mult(comm, nullptr, &A, idx_A.idx, &B.tensor_, idx_B.idx, &C, idx_C.idx);
}

void mult(const communicator& comm,
          const scalar& alpha,
          const const_tensor& A,
          const const_tensor& B,
          const scalar& beta,
          const tensor& C)
{
    label_vector idx_A, idx_B, idx_C;

    TBLIS_ASSERT((A.tensor_.ndim+B.tensor_.ndim+C.ndim)%2 == 0);

    auto nAB = (A.tensor_.ndim+B.tensor_.ndim-C.ndim)/2;
    auto nAC = (A.tensor_.ndim+C.ndim-B.tensor_.ndim)/2;
    auto nBC = (B.tensor_.ndim+C.ndim-A.tensor_.ndim)/2;

    for (auto i : range(nAC)) idx_A.push_back(i);
    for (auto i : range(nAC)) idx_C.push_back(i);
    for (auto i : range(nAB)) idx_A.push_back(nAC+i);
    for (auto i : range(nAB)) idx_B.push_back(nAC+i);
    for (auto i : range(nBC)) idx_B.push_back(nAC+nAB+i);
    for (auto i : range(nBC)) idx_C.push_back(nAC+nAB+i);

    mult(comm, alpha, A, idx_A, B, idx_B, beta, C, idx_C);
}

}
