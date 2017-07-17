#include "mult.h"

#include "util/macros.h"
#include "util/tensor.hpp"
#include "internal/1t/dense/scale.hpp"
#include "internal/1t/dense/set.hpp"
#include "internal/3t/dense/mult.hpp"
#include "internal/1t/dpd/scale.hpp"
#include "internal/1t/dpd/set.hpp"
#include "internal/3t/dpd/mult.hpp"
#include "internal/1t/indexed/scale.hpp"
#include "internal/1t/indexed/set.hpp"
#include "internal/3t/indexed/mult.hpp"
#include "internal/1t/indexed_dpd/scale.hpp"
#include "internal/1t/indexed_dpd/set.hpp"
#include "internal/3t/indexed_dpd/mult.hpp"

namespace tblis
{

extern "C"
{

void tblis_tensor_mult(const tblis_comm* comm, const tblis_config* cfg,
                       const tblis_tensor* A, const label_type* idx_A_,
                       const tblis_tensor* B, const label_type* idx_B_,
                             tblis_tensor* C, const label_type* idx_C_)
{
    TBLIS_ASSERT(A->type == B->type);
    TBLIS_ASSERT(A->type == C->type);

    unsigned ndim_A = A->ndim;
    len_vector len_A;
    stride_vector stride_A;
    label_vector idx_A;
    diagonal(ndim_A, A->len, A->stride, idx_A_, len_A, stride_A, idx_A);

    unsigned ndim_B = B->ndim;
    len_vector len_B;
    stride_vector stride_B;
    label_vector idx_B;
    diagonal(ndim_B, B->len, B->stride, idx_B_, len_B, stride_B, idx_B);

    unsigned ndim_C = C->ndim;
    len_vector len_C;
    stride_vector stride_C;
    label_vector idx_C;
    diagonal(ndim_C, C->len, C->stride, idx_C_, len_C, stride_C, idx_C);

    if (idx_A.empty() || idx_B.empty() || idx_C.empty())
    {
        len_A.push_back(1);
        len_B.push_back(1);
        len_C.push_back(1);
        stride_A.push_back(0);
        stride_B.push_back(0);
        stride_C.push_back(0);
        label_type idx = detail::free_idx(idx_A, idx_B, idx_C);
        idx_A.push_back(idx);
        idx_B.push_back(idx);
        idx_C.push_back(idx);
    }

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

    fold(len_ABC, idx_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);
    fold(len_AB, idx_AB, stride_A_AB, stride_B_AB);
    fold(len_AC, idx_AC, stride_A_AC, stride_C_AC);
    fold(len_BC, idx_BC, stride_B_BC, stride_C_BC);

    len_vector nolen;
    stride_vector nostride;

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        T alpha = A->alpha<T>()*B->alpha<T>();
        T beta = C->alpha<T>();

        T* data_A = static_cast<T*>(A->data);
        T* data_B = static_cast<T*>(B->data);
        T* data_C = static_cast<T*>(C->data);

        parallelize_if(
        [&](const communicator& comm)
        {
            if (alpha == T(0))
            {
                if (beta == T(0))
                {
                    internal::set<T>(comm, get_config(cfg),
                                     len_AC+len_BC+len_ABC, T(0), data_C,
                                     stride_C_AC+stride_C_BC+stride_C_ABC);
                }
                else if (beta != T(1) || (is_complex<T>::value && C->conj))
                {
                    internal::scale<T>(comm, get_config(cfg),
                                       len_AC+len_BC+len_ABC,
                                       beta, C->conj, data_C,
                                       stride_C_AC+stride_C_BC+stride_C_ABC);
                }
            }
            else
            {
                internal::mult<T>(comm, get_config(cfg),
                                  len_AB, len_AC, len_BC, len_ABC,
                                  alpha, A->conj, data_A,
                                  stride_A_AB, stride_A_AC, stride_A_ABC,
                                         B->conj, data_B,
                                  stride_B_AB, stride_B_BC, stride_B_ABC,
                                   beta, C->conj, data_C,
                                  stride_C_AC, stride_C_BC, stride_C_ABC);
            }
        }, comm);

        C->alpha<T>() = T(1);
        C->conj = false;
    })
}

}

template <typename T>
void mult(const communicator& comm,
          T alpha, dpd_varray_view<const T> A, const label_type* idx_A_,
                   dpd_varray_view<const T> B, const label_type* idx_B_,
          T  beta, dpd_varray_view<      T> C, const label_type* idx_C_)
{
    unsigned nirrep = A.num_irreps();
    TBLIS_ASSERT(B.num_irreps() == nirrep);
    TBLIS_ASSERT(C.num_irreps() == nirrep);

    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();
    unsigned ndim_C = C.dimension();

    std::string idx_A(idx_A_, idx_A_+ndim_A);
    std::string idx_B(idx_B_, idx_B_+ndim_B);
    std::string idx_C(idx_C_, idx_C_+ndim_C);

    for (unsigned i = 1;i < ndim_A;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    for (unsigned i = 1;i < ndim_B;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_B[i] != idx_B[j]);

    for (unsigned i = 1;i < ndim_C;i++)
        for (unsigned j = 0;j < i;j++)
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

    for (unsigned i = 0;i < idx_ABC.size();i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            TBLIS_ASSERT(A.length(idx_A_ABC[i], irrep) ==
                         B.length(idx_B_ABC[i], irrep));
            TBLIS_ASSERT(A.length(idx_A_ABC[i], irrep) ==
                         C.length(idx_C_ABC[i], irrep));
        }
    }

    for (unsigned i = 0;i < idx_AB.size();i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            TBLIS_ASSERT(A.length(idx_A_AB[i], irrep) ==
                         B.length(idx_B_AB[i], irrep));
        }
    }

    for (unsigned i = 0;i < idx_AC.size();i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            TBLIS_ASSERT(A.length(idx_A_AC[i], irrep) ==
                         C.length(idx_C_AC[i], irrep));
        }
    }

    for (unsigned i = 0;i < idx_BC.size();i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            TBLIS_ASSERT(B.length(idx_B_BC[i], irrep) ==
                         C.length(idx_C_BC[i], irrep));
        }
    }

    if (alpha == T(0) || (idx_ABC.empty() && (A.irrep()^B.irrep() != C.irrep())))
    {
        if (beta == T(0))
        {
            internal::set(comm, get_default_config(),
                          beta, C, range_C);
        }
        else if (beta != T(1))
        {
            internal::scale(comm, get_default_config(),
                            beta, false, C, range_C);
        }
    }
    else
    {
        internal::mult(comm, get_default_config(),
                       alpha, false, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                              false, B, idx_B_AB, idx_B_BC, idx_B_ABC,
                        beta, false, C, idx_C_AC, idx_C_BC, idx_C_ABC);
    }
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, \
                   T alpha, dpd_varray_view<const T> A, const label_type* idx_A, \
                            dpd_varray_view<const T> B, const label_type* idx_B, \
                   T  beta, dpd_varray_view<      T> C, const label_type* idx_C);
#include "configs/foreach_type.h"

template <typename T>
void mult(const communicator& comm,
          T alpha, indexed_varray_view<const T> A, const label_type* idx_A_,
                   indexed_varray_view<const T> B, const label_type* idx_B_,
          T  beta, indexed_varray_view<      T> C, const label_type* idx_C_)
{
    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();
    unsigned ndim_C = C.dimension();

    std::string idx_A(idx_A_, idx_A_+ndim_A);
    std::string idx_B(idx_B_, idx_B_+ndim_B);
    std::string idx_C(idx_C_, idx_C_+ndim_C);

    for (unsigned i = 1;i < ndim_A;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    for (unsigned i = 1;i < ndim_B;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_B[i] != idx_B[j]);

    for (unsigned i = 1;i < ndim_C;i++)
        for (unsigned j = 0;j < i;j++)
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

    for (unsigned i = 0;i < idx_ABC.size();i++)
    {
        TBLIS_ASSERT(A.length(idx_A_ABC[i]) ==
                     B.length(idx_B_ABC[i]));
        TBLIS_ASSERT(A.length(idx_A_ABC[i]) ==
                     C.length(idx_C_ABC[i]));
    }

    for (unsigned i = 0;i < idx_AB.size();i++)
    {
        TBLIS_ASSERT(A.length(idx_A_AB[i]) ==
                     B.length(idx_B_AB[i]));
    }

    for (unsigned i = 0;i < idx_AC.size();i++)
    {
        TBLIS_ASSERT(A.length(idx_A_AC[i]) ==
                     C.length(idx_C_AC[i]));
    }

    for (unsigned i = 0;i < idx_BC.size();i++)
    {
        TBLIS_ASSERT(B.length(idx_B_BC[i]) ==
                     C.length(idx_C_BC[i]));
    }

    if (alpha == T(0))
    {
        if (beta == T(0))
        {
            internal::set(comm, get_default_config(),
                          beta, C, range_C);
        }
        else if (beta != T(1))
        {
            internal::scale(comm, get_default_config(),
                            beta, false, C, range_C);
        }
    }
    else
    {
        internal::mult(comm, get_default_config(),
                       alpha, false, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                              false, B, idx_B_AB, idx_B_BC, idx_C_ABC,
                        beta, false, C, idx_C_AC, idx_C_BC, idx_B_ABC);
    }
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, \
                   T alpha, indexed_varray_view<const T> A, const label_type* idx_A, \
                            indexed_varray_view<const T> B, const label_type* idx_B, \
                   T  beta, indexed_varray_view<      T> C, const label_type* idx_C);
#include "configs/foreach_type.h"

template <typename T>
void mult(const communicator& comm,
          T alpha, indexed_dpd_varray_view<const T> A, const label_type* idx_A_,
                   indexed_dpd_varray_view<const T> B, const label_type* idx_B_,
          T  beta, indexed_dpd_varray_view<      T> C, const label_type* idx_C_)
{
    unsigned nirrep = A.num_irreps();
    TBLIS_ASSERT(B.num_irreps() == nirrep);
    TBLIS_ASSERT(C.num_irreps() == nirrep);

    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();
    unsigned ndim_C = C.dimension();

    std::string idx_A(idx_A_, idx_A_+ndim_A);
    std::string idx_B(idx_B_, idx_B_+ndim_B);
    std::string idx_C(idx_C_, idx_C_+ndim_C);

    for (unsigned i = 1;i < ndim_A;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    for (unsigned i = 1;i < ndim_B;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_B[i] != idx_B[j]);

    for (unsigned i = 1;i < ndim_C;i++)
        for (unsigned j = 0;j < i;j++)
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

    for (unsigned i = 0;i < idx_ABC.size();i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            TBLIS_ASSERT(A.length(idx_A_ABC[i], irrep) ==
                         B.length(idx_B_ABC[i], irrep));
            TBLIS_ASSERT(A.length(idx_A_ABC[i], irrep) ==
                         C.length(idx_C_ABC[i], irrep));
        }
    }

    for (unsigned i = 0;i < idx_AB.size();i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            TBLIS_ASSERT(A.length(idx_A_AB[i], irrep) ==
                         B.length(idx_B_AB[i], irrep));
        }
    }

    for (unsigned i = 0;i < idx_AC.size();i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            TBLIS_ASSERT(A.length(idx_A_AC[i], irrep) ==
                         C.length(idx_C_AC[i], irrep));
        }
    }

    for (unsigned i = 0;i < idx_BC.size();i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            TBLIS_ASSERT(B.length(idx_B_BC[i], irrep) ==
                         C.length(idx_C_BC[i], irrep));
        }
    }

    if (alpha == T(0) || (idx_ABC.empty() && (A.irrep()^B.irrep() != C.irrep())))
    {
        if (beta == T(0))
        {
            internal::set(comm, get_default_config(),
                          beta, C, range_C);
        }
        else if (beta != T(1))
        {
            internal::scale(comm, get_default_config(),
                            beta, false, C, range_C);
        }
    }
    else
    {
        internal::mult(comm, get_default_config(),
                       alpha, false, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                              false, B, idx_B_AB, idx_B_BC, idx_C_ABC,
                        beta, false, C, idx_C_AC, idx_C_BC, idx_B_ABC);
    }
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, \
                   T alpha, indexed_dpd_varray_view<const T> A, const label_type* idx_A, \
                            indexed_dpd_varray_view<const T> B, const label_type* idx_B, \
                   T  beta, indexed_dpd_varray_view<      T> C, const label_type* idx_C);
#include "configs/foreach_type.h"

}
