#include "util.hpp"
#include "dot.hpp"
#include "internal/1t/dense/dot.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dot_full(const communicator& comm, const config& cfg,
              bool conj_A, const indexed_varray_view<const T>& A,
              const dim_vector& idx_A_AB,
              bool conj_B, const indexed_varray_view<const T>& B,
              const dim_vector& idx_B_AB,
              T& result)
{
    varray<T> A2, B2;

    comm.broadcast(
    [&](varray<T>& A2, varray<T>& B2)
    {
        block_to_full(comm, cfg, A, A2);
        block_to_full(comm, cfg, B, B2);

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
               bool conj_A, const indexed_varray_view<const T>& A,
               const dim_vector& idx_A_AB,
               bool conj_B, const indexed_varray_view<const T>& B,
               const dim_vector& idx_B_AB,
               T& result)
{
    index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);

    group_indices<T, 1> indices_A(A, group_AB, 0);
    group_indices<T, 1> indices_B(B, group_AB, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();

    atomic_accumulator<T> local_result;

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    comm.do_tasks_deferred(std::min(nidx_A, nidx_B),
                           stl_ext::prod(group_AB.dense_len)*inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        for_each_match<false, false>(idx_A, nidx_A, indices_A, 0,
                                    idx_B, nidx_B, indices_B, 0,
        [&]
        {
            auto factor = indices_A[idx_A].factor*indices_B[idx_B].factor;
            if (factor == T(0)) return;

            tasks.visit(idx++,
            [&,idx_A,idx_B,factor](const communicator& subcomm)
            {
                stride_type off_A_AB, off_B_AB;
                get_local_offset(indices_A[idx_A].idx[0], group_AB,
                                 off_A_AB, 0, off_B_AB, 1);

                auto data_A = A.data(0) + indices_A[idx_A].offset + off_A_AB;
                auto data_B = B.data(0) + indices_B[idx_B].offset + off_B_AB;

                T block_result;
                dot(subcomm, cfg, group_AB.dense_len,
                    conj_A, data_A, group_AB.dense_stride[0],
                    conj_B, data_B, group_AB.dense_stride[1],
                    block_result);

                if (subcomm.master()) local_result += factor*block_result;
            });
        });
    });

    reduce(comm, local_result);
    if (comm.master()) result = local_result;
}

template <typename T>
void dot(const communicator& comm, const config& cfg,
         bool conj_A, const indexed_varray_view<const T>& A,
         const dim_vector& idx_A_AB,
         bool conj_B, const indexed_varray_view<const T>& B,
         const dim_vector& idx_B_AB,
         T& result)
{
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
                  bool conj_A, const indexed_varray_view<const T>& A, \
                  const dim_vector& idx_A_AB, \
                  bool conj_B, const indexed_varray_view<const T>& B, \
                  const dim_vector& idx_B_AB, \
                  T& result);
#include "configs/foreach_type.h"

}
}
