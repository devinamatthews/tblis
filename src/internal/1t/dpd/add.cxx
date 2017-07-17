#include "util.hpp"
#include "add.hpp"
#include "internal/1t/dense/add.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void add_full(const communicator& comm, const config& cfg,
               T alpha, bool conj_A, const dpd_varray_view<const T>& A,
               const dim_vector& idx_A_A,
               const dim_vector& idx_A_AB,
               T  beta, bool conj_B, const dpd_varray_view<      T>& B,
               const dim_vector& idx_B_B,
               const dim_vector& idx_B_AB)
{
    varray<T> A2, B2;

    comm.broadcast(
    [&](varray<T>& A2, varray<T>& B2)
    {
        if (comm.master())
        {
            block_to_full(A, A2);
            block_to_full(B, B2);
        }

        auto len_A = stl_ext::select_from(A2.lengths(), idx_A_A);
        auto len_B = stl_ext::select_from(B2.lengths(), idx_B_B);
        auto len_AB = stl_ext::select_from(A2.lengths(), idx_A_AB);
        auto stride_A_A = stl_ext::select_from(A2.strides(), idx_A_A);
        auto stride_B_B = stl_ext::select_from(B2.strides(), idx_B_B);
        auto stride_A_AB = stl_ext::select_from(A2.strides(), idx_A_AB);
        auto stride_B_AB = stl_ext::select_from(B2.strides(), idx_B_AB);

        add(comm, cfg, len_A, len_B, len_AB,
            alpha, conj_A, A2.data(), stride_A_A, stride_A_AB,
             beta, conj_B, B2.data(), stride_B_B, stride_B_AB);

        if (comm.master())
        {
            full_to_block(B2, B);
        }
    },
    A2, B2);
}

template <typename T>
void trace_block(const communicator& comm, const config& cfg,
                 T alpha, bool conj_A, const dpd_varray_view<const T>& A,
                 const dim_vector& idx_A_A,
                 const dim_vector& idx_A_AB,
                 T  beta, bool conj_B, const dpd_varray_view<      T>& B,
                 const dim_vector& idx_B_AB)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();

    unsigned ndim_A_only = idx_A_A.size();
    unsigned ndim_AB = idx_A_AB.size();

    len_vector len_A(ndim_A);
    for (unsigned i = 0;i < ndim_A;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            len_A[i] += A.length(i, irrep);
    }

    len_vector len_B(ndim_B);
    for (unsigned i = 0;i < ndim_B;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            len_B[i] += B.length(i, irrep);
    }

    stride_type dense_A = 1;
    stride_type nblock_A = 1;
    for (unsigned i : idx_A_A)
    {
        dense_A *= len_A[i];
        nblock_A *= nirrep;
    }
    dense_A /= nblock_A;

    stride_type dense_AB = 1;
    stride_type nblock_AB = 1;
    for (unsigned i : idx_A_AB)
    {
        dense_AB *= len_A[i];
        nblock_AB *= nirrep;
    }
    dense_AB /= nblock_AB;

    if (nblock_A > 1) nblock_A /= nirrep;
    if (nblock_AB > 1) nblock_AB /= nirrep;

    irrep_vector irreps_A(ndim_A);
    irrep_vector irreps_B(ndim_B);

    unsigned irrep_AB = B.irrep();
    unsigned irrep_A = A.irrep()^irrep_AB;

    for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
    {
        assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
              irreps_A, idx_A_AB, irreps_B, idx_B_AB);

        if (is_block_empty(B, irreps_B)) continue;

        auto local_B = B(irreps_B);

        auto len_AB = stl_ext::select_from(local_B.lengths(), idx_B_AB);
        auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

        for (stride_type block_A = 0;block_A < nblock_A;block_A++)
        {
            assign_irreps(ndim_A_only, irrep_A, nirrep, block_A,
                  irreps_A, idx_A_A);

            if (is_block_empty(A, irreps_A)) continue;

            auto local_A = A(irreps_A);

            auto len_A_only = stl_ext::select_from(local_A.lengths(), idx_A_A);
            auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
            auto stride_A_A = stl_ext::select_from(local_A.strides(), idx_A_A);

            add(comm, cfg, len_A_only, {}, len_AB,
                alpha, conj_A, local_A.data(), stride_A_A, stride_A_AB,
                 beta, conj_B, local_B.data(), {}, stride_B_AB);

            beta = T(1);
        }
    }
}

template <typename T>
void replicate_block(const communicator& comm, const config& cfg,
                     T alpha, bool conj_A, const dpd_varray_view<const T>& A,
                     const dim_vector& idx_A_AB,
                     T  beta, bool conj_B, const dpd_varray_view<      T>& B,
                     const dim_vector& idx_B_B,
                     const dim_vector& idx_B_AB)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();

    unsigned ndim_B_only = idx_B_B.size();
    unsigned ndim_AB = idx_A_AB.size();

    len_vector len_A(ndim_A);
    for (unsigned i = 0;i < ndim_A;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            len_A[i] += A.length(i, irrep);
    }

    len_vector len_B(ndim_B);
    for (unsigned i = 0;i < ndim_B;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            len_B[i] += B.length(i, irrep);
    }

    stride_type dense_B = 1;
    stride_type nblock_B = 1;
    for (unsigned i : idx_B_B)
    {
        dense_B *= len_B[i];
        nblock_B *= nirrep;
    }
    dense_B /= nblock_B;

    stride_type dense_AB = 1;
    stride_type nblock_AB = 1;
    for (unsigned i : idx_A_AB)
    {
        dense_AB *= len_A[i];
        nblock_AB *= nirrep;
    }
    dense_AB /= nblock_AB;

    if (nblock_B > 1) nblock_B /= nirrep;
    if (nblock_AB > 1) nblock_AB /= nirrep;

    irrep_vector irreps_A(ndim_A);
    irrep_vector irreps_B(ndim_B);

    unsigned irrep_AB = A.irrep();
    unsigned irrep_B = B.irrep()^irrep_AB;

    for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
    {
        assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
              irreps_A, idx_A_AB, irreps_B, idx_B_AB);

        if (is_block_empty(A, irreps_A)) continue;

        auto local_A = A(irreps_A);

        auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
        auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);

        for (stride_type block_B = 0;block_B < nblock_B;block_B++)
        {
            assign_irreps(ndim_B_only, irrep_B, nirrep, block_B,
                  irreps_B, idx_B_B);

            if (is_block_empty(B, irreps_B)) continue;

            auto local_B = B(irreps_B);

            auto len_B_only = stl_ext::select_from(local_B.lengths(), idx_B_B);
            auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);
            auto stride_B_B = stl_ext::select_from(local_B.strides(), idx_B_B);

            add(comm, cfg, {}, len_B_only, len_AB,
                alpha, conj_A, local_A.data(), {}, stride_A_AB,
                 beta, conj_B, local_B.data(), stride_B_B, stride_B_AB);
        }
    }
}

template <typename T>
void transpose_block(const communicator& comm, const config& cfg,
                     T alpha, bool conj_A, const dpd_varray_view<const T>& A,
                     const dim_vector& idx_A_AB,
                     T  beta, bool conj_B, const dpd_varray_view<      T>& B,
                     const dim_vector& idx_B_AB)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();

    unsigned ndim_AB = idx_A_AB.size();

    len_vector len_A(ndim_A);
    for (unsigned i = 0;i < ndim_A;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            len_A[i] += A.length(i, irrep);
    }

    len_vector len_B(ndim_B);
    for (unsigned i = 0;i < ndim_B;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            len_B[i] += B.length(i, irrep);
    }

    stride_type dense_AB = 1;
    stride_type nblock_AB = 1;
    for (unsigned i : idx_A_AB)
    {
        dense_AB *= len_A[i];
        nblock_AB *= nirrep;
    }
    dense_AB /= nblock_AB;

    if (nblock_AB > 1) nblock_AB /= nirrep;

    irrep_vector irreps_A(ndim_A);
    irrep_vector irreps_B(ndim_B);

    unsigned irrep_AB = A.irrep();

    for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
    {
        assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
              irreps_A, idx_A_AB, irreps_B, idx_B_AB);

        if (is_block_empty(A, irreps_A)) continue;

        auto local_A = A(irreps_A);
        auto local_B = B(irreps_B);

        auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
        auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
        auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

        add(comm, cfg, {}, {}, len_AB,
            alpha, conj_A, local_A.data(), {}, stride_A_AB,
             beta, conj_B, local_B.data(), {}, stride_B_AB);
    }
}

template <typename T>
void add(const communicator& comm, const config& cfg,
         T alpha, bool conj_A, const dpd_varray_view<const T>& A,
         const dim_vector& idx_A_A,
         const dim_vector& idx_A_AB,
         T  beta, bool conj_B, const dpd_varray_view<      T>& B,
         const dim_vector& idx_B_B,
         const dim_vector& idx_B_AB)
{
    if (dpd_impl == FULL)
    {
        add_full(comm, cfg,
                 alpha, conj_A, A, idx_A_A, idx_A_AB,
                  beta, conj_B, B, idx_B_B, idx_B_AB);
    }
    else if (!idx_A_A.empty())
    {
        trace_block(comm, cfg,
                    alpha, conj_A, A, idx_A_A, idx_A_AB,
                     beta, conj_B, B, idx_B_AB);
    }
    else if (!idx_B_B.empty())
    {
        replicate_block(comm, cfg,
                        alpha, conj_A, A, idx_A_AB,
                         beta, conj_B, B, idx_B_B, idx_B_AB);
    }
    else
    {
        transpose_block(comm, cfg,
                        alpha, conj_A, A, idx_A_AB,
                         beta, conj_B, B, idx_B_AB);
    }
}

#define FOREACH_TYPE(T) \
template void add(const communicator& comm, const config& cfg, \
                  T alpha, bool conj_A, const dpd_varray_view<const T>& A, \
                  const dim_vector& idx_A, \
                  const dim_vector& idx_A_AB, \
                  T  beta, bool conj_B, const dpd_varray_view<      T>& B, \
                  const dim_vector& idx_B, \
                  const dim_vector& idx_B_AB);
#include "configs/foreach_type.h"

}
}
