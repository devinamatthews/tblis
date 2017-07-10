#include "dpd_add.hpp"
#include "add.hpp"

#include "util/tensor.hpp"

#include "external/stl_ext/include/iostream.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dpd_add(const communicator& comm, const config& cfg,
             T alpha, bool conj_A, const dpd_varray_view<const T>& A,
             const std::vector<unsigned>& idx_A_A,
             const std::vector<unsigned>& idx_A_AB,
             T  beta, bool conj_B, const dpd_varray_view<      T>& B,
             const std::vector<unsigned>& idx_B_B,
             const std::vector<unsigned>& idx_B_AB)
{
    using stl_ext::intersection;
    using stl_ext::exclusion;
    using stl_ext::select_from;
    using stl_ext::permute;

    unsigned nirrep = A.num_irreps();

    if (!idx_A_A.empty())
    {
        unsigned ndim_A = A.dimension();
        unsigned ndim_B = B.dimension();

        unsigned ndim_A_only = idx_A_A.size();
        unsigned ndim_AB = idx_A_AB.size();

        std::vector<len_type> len_A(ndim_A);
        for (unsigned i = 0;i < ndim_A;i++)
        {
            for (unsigned irrep = 0;irrep < nirrep;irrep++)
                len_A[i] += A.length(i, irrep);
        }

        std::vector<len_type> len_B(ndim_B);
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

        std::vector<unsigned> irreps_A(ndim_A);
        std::vector<unsigned> irreps_B(ndim_B);

        unsigned irrep_AB = B.irrep();
        unsigned irrep_A = A.irrep()^irrep_AB;

        if (ndim_A_only == 0 && irrep_A != 0) continue;

        for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
        {
            detail::assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
                          irreps_A, idx_A_AB, irreps_B, idx_B_AB);

            auto local_B = B(irreps_B);

            auto len_AB = stl_ext::select_from(local_B.lengths(), idx_B_AB);
            auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

            for (stride_type block_A = 0;block_A < nblock_A;block_A++)
            {
                detail::assign_irreps(ndim_A_only, irrep_A, nirrep, block_A,
                              irreps_A, idx_A_A);

                auto local_A = A(irreps_A);

                auto len_A_only = stl_ext::select_from(local_A.lengths(), idx_A_A);
                auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
                auto stride_A_A = stl_ext::select_from(local_A.strides(), idx_A_A);

                internal::add<T>(comm, cfg, len_A_only, {}, len_AB,
                                 alpha, conj_A, local_A.data(),
                                 stride_A_A, stride_A_AB,
                                 beta, conj_B, local_B.data(),
                                 {}, stride_B_AB);

                beta = T(1);
            }
        }
    }
    else if (!idx_B_B.empty())
    {
        unsigned ndim_A = A.dimension();
        unsigned ndim_B = B.dimension();

        unsigned ndim_B_only = idx_B_B.size();
        unsigned ndim_AB = idx_A_AB.size();

        std::vector<len_type> len_A(ndim_A);
        for (unsigned i = 0;i < ndim_A;i++)
        {
            for (unsigned irrep = 0;irrep < nirrep;irrep++)
                len_A[i] += A.length(i, irrep);
        }

        std::vector<len_type> len_B(ndim_B);
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

        std::vector<unsigned> irreps_A(ndim_A);
        std::vector<unsigned> irreps_B(ndim_B);

        unsigned irrep_AB = A.irrep();
        unsigned irrep_B = B.irrep()^irrep_AB;

        if (ndim_B_only == 0 && irrep_B != 0) continue;

        for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
        {
            detail::assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
                          irreps_A, idx_A_AB, irreps_B, idx_B_AB);

            auto local_A = A(irreps_A);

            auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
            auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);

            for (stride_type block_B = 0;block_B < nblock_B;block_B++)
            {
                detail::assign_irreps(ndim_B_only, irrep_B, nirrep, block_B,
                              irreps_B, idx_B_B);

                auto local_B = B(irreps_B);

                auto len_B_only = stl_ext::select_from(local_B.lengths(), idx_B_B);
                auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);
                auto stride_B_B = stl_ext::select_from(local_B.strides(), idx_B_B);

                internal::add<T>(comm, cfg, {}, len_B_only, len_AB,
                                 alpha, conj_A, local_A.data(),
                                 {}, stride_A_AB,
                                 beta, conj_B, local_B.data(),
                                 stride_B_B, stride_B_AB);
            }
        }
    }
    else
    {
        unsigned ndim_A = A.dimension();
        unsigned ndim_B = B.dimension();

        unsigned ndim_AB = idx_A_AB.size();

        std::vector<len_type> len_A(ndim_A);
        for (unsigned i = 0;i < ndim_A;i++)
        {
            for (unsigned irrep = 0;irrep < nirrep;irrep++)
                len_A[i] += A.length(i, irrep);
        }

        std::vector<len_type> len_B(ndim_B);
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

        std::vector<unsigned> irreps_A(ndim_A);
        std::vector<unsigned> irreps_B(ndim_B);

        unsigned irrep_AB = A.irrep();

        for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
        {
            detail::assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
                          irreps_A, idx_A_AB, irreps_B, idx_B_AB);

            auto local_A = A(irreps_A);
            auto local_B = B(irreps_B);

            auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
            auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
            auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

            internal::add<T>(comm, cfg, {}, {}, len_AB,
                             alpha, conj_A, local_A.data(),
                             {}, stride_A_AB,
                             beta, conj_B, local_B.data(),
                             {}, stride_B_AB);
        }
    }
}

#define FOREACH_TYPE(T) \
template void dpd_add(const communicator& comm, const config& cfg, \
                      T alpha, bool conj_A, const dpd_varray_view<const T>& A, \
                      const std::vector<unsigned>& idx_A, \
                      const std::vector<unsigned>& idx_A_AB, \
                      T  beta, bool conj_B, const dpd_varray_view<      T>& B, \
                      const std::vector<unsigned>& idx_B, \
                      const std::vector<unsigned>& idx_B_AB);
#include "configs/foreach_type.h"

}
}
