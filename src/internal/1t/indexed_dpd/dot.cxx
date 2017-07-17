#include "util.hpp"
#include "dot.hpp"
#include "internal/1t/dense/dot.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dot_full(const communicator& comm, const config& cfg,
              bool conj_A, const indexed_dpd_varray_view<const T>& A,
              const dim_vector& idx_A_AB,
              bool conj_B, const indexed_dpd_varray_view<const T>& B,
              const dim_vector& idx_B_AB,
              T& result)
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

        auto len_AB = stl_ext::select_from(A2.lengths(), idx_A_AB);
        auto stride_A_AB = stl_ext::select_from(A2.strides(), idx_A_AB);
        auto stride_B_AB = stl_ext::select_from(B2.strides(), idx_B_AB);

        dot(comm, cfg, len_AB,
            conj_A, A2.data(), stride_A_AB,
            conj_B, B2.data(), stride_B_AB,
            result);
    },
    A2, B2);
}

template <typename T>
void dot_block(const communicator& comm, const config& cfg,
               bool conj_A, const indexed_dpd_varray_view<const T>& A,
               const dim_vector& idx_A_AB,
               bool conj_B, const indexed_dpd_varray_view<const T>& B,
               const dim_vector& idx_B_AB,
               T& result)
{
    unsigned nirrep = A.num_irreps();

    dpd_index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);

    irrep_vector irreps_A(A.dense_dimension());
    irrep_vector irreps_B(B.dense_dimension());
    assign_irreps(group_AB, irreps_A, irreps_B);

    unsigned irrep_AB = A.irrep();
    for (auto irrep : group_AB.batch_irrep) irrep_AB ^= irrep;

    group_indices<1> indices_A(A, group_AB, 0);
    group_indices<1> indices_B(B, group_AB, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();

    auto dpd_A = A[0];
    auto dpd_B = B[0];

    dynamic_task_set tasks(comm, nidx_B*group_AB.dense_nblock, group_AB.dense_size);

    stride_type idx_A = 0;
    stride_type idx_B = 0;

    len_vector len_AB;
    stride_vector stride_A_AB, stride_B_AB;
    stride_type off_A_AB, off_B_AB;

    T local_result = T();
    T block_result;

    while (idx_A < nidx_A && idx_B < nidx_B)
    {
        if (indices_A[idx_A].key[0] < indices_B[idx_B].key[0])
        {
            idx_A++;
        }
        else if (indices_A[idx_A].key[0] > indices_B[idx_B].key[0])
        {
            idx_B++;
        }
        else
        {
            for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
            {
                assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                      irreps_A, group_AB.dense_idx[0],
                                      irreps_B, group_AB.dense_idx[1]);

                if (is_block_empty(dpd_A, irreps_A)) continue;
                if (is_block_empty(dpd_B, irreps_B)) continue;

                auto local_A = dpd_A(irreps_A);
                auto local_B = dpd_B(irreps_B);

                get_local_geometry(indices_A[idx_A].idx[0], group_AB, len_AB,
                                   local_A, off_A_AB, stride_A_AB, 0,
                                   local_B, off_B_AB, stride_B_AB, 1);

                auto data_A = local_A.data() + indices_A[idx_A].offset + off_A_AB;
                auto data_B = local_B.data() + indices_B[idx_B].offset + off_B_AB;

                dot(comm, cfg, len_AB,
                    conj_A, data_A, stride_A_AB,
                    conj_B, data_B, stride_B_AB,
                    block_result);

                local_result += block_result;
            }

            idx_A++;
            idx_B++;
        }
    }

    if (comm.master()) result = local_result;
}

template <typename T>
void dot(const communicator& comm, const config& cfg,
         bool conj_A, const indexed_dpd_varray_view<const T>& A,
         const dim_vector& idx_A_AB,
         bool conj_B, const indexed_dpd_varray_view<const T>& B,
         const dim_vector& idx_B_AB,
         T& result)
{
    if (A.irrep() != B.irrep())
    {
        if (comm.master()) result = 0;
        comm.barrier();
        return;
    }

    if (dpd_impl == FULL)
    {
        dot_full(comm, cfg,
                 conj_A, A, idx_A_AB,
                 conj_B, B, idx_B_AB,
                 result);
    }
    else
    {
        dot_block(comm, cfg,
                  conj_A, A, idx_A_AB,
                  conj_B, B, idx_B_AB,
                  result);
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void dot(const communicator& comm, const config& cfg, \
                  bool conj_A, const indexed_dpd_varray_view<const T>& A, \
                  const dim_vector& idx_A_AB, \
                  bool conj_B, const indexed_dpd_varray_view<const T>& B, \
                  const dim_vector& idx_B_AB, \
                  T& result);
#include "configs/foreach_type.h"

}
}
