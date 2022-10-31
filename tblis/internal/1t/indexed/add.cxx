#include "util.hpp"
#include "add.hpp"
#include "scale.hpp"
#include "set.hpp"
#include "internal/1t/dense/add.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void add_full(const communicator& comm, const config& cfg,
              T alpha, bool conj_A, const indexed_marray_view<T>& A,
              const dim_vector& idx_A_A,
              const dim_vector& idx_A_AB,
                                    const indexed_marray_view<T>& B,
              const dim_vector& idx_B_B,
              const dim_vector& idx_B_AB)
{
    marray<T> A2, B2;

    comm.broadcast(
    [&](marray<T>& A2, marray<T>& B2)
    {
        block_to_full(comm, cfg, A, A2);
        block_to_full(comm, cfg, B, B2);

        auto len_A = stl_ext::select_from(A2.lengths(), idx_A_A);
        auto len_B = stl_ext::select_from(B2.lengths(), idx_B_B);
        auto len_AB = stl_ext::select_from(A2.lengths(), idx_A_AB);
        auto stride_A_A = stl_ext::select_from(A2.strides(), idx_A_A);
        auto stride_B_B = stl_ext::select_from(B2.strides(), idx_B_B);
        auto stride_A_AB = stl_ext::select_from(A2.strides(), idx_A_AB);
        auto stride_B_AB = stl_ext::select_from(B2.strides(), idx_B_AB);

        add(type_tag<T>::value, comm, cfg, len_A, len_B, len_AB,
            alpha, conj_A, reinterpret_cast<char*>(A2.data()), stride_A_A, stride_A_AB,
             T(0),  false, reinterpret_cast<char*>(B2.data()), stride_B_B, stride_B_AB);

        full_to_block(comm, cfg, B2, B);
    },
    A2, B2);
}

void trace_block(type_t type, const communicator& comm, const config& cfg,
                 const scalar& alpha, bool conj_A,
                 const indexed_marray_view<char>& A,
                 const dim_vector& idx_A_A,
                 const dim_vector& idx_A_AB,
                 const indexed_marray_view<char>& B,
                 const dim_vector& idx_B_AB)
{
    const len_type ts = type_size[type];

    index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    index_group<1> group_A(A, idx_A_A);

    group_indices<2> indices_A(type, A, group_AB, 0, group_A, 0);
    group_indices<1> indices_B(type, B, group_AB, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    scalar one(1.0, type);

    comm.do_tasks_deferred(nidx_B, stl_ext::prod(group_AB.dense_len)*
                                   stl_ext::prod(group_A.dense_len)*inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        for_each_match<true, false>(idx_A, nidx_A, indices_A, 0,
                                    idx_B, nidx_B, indices_B, 0,
        [&](stride_type next_A)
        {
            if (indices_B[idx_B].factor.is_zero()) return;

            tasks.visit(idx++,
            [&,idx_A,idx_B,next_A](const communicator& subcomm)
            {
                stride_type off_A_AB, off_B_AB;
                get_local_offset(indices_A[idx_A].idx[0], group_AB,
                                 off_A_AB, 0, off_B_AB, 1);

                auto data_B = B.data(0) + (indices_B[idx_B].offset + off_B_AB)*ts;

                for (auto local_idx_A = idx_A;local_idx_A < next_A;local_idx_A++)
                {
                    auto factor = alpha*indices_A[local_idx_A].factor*
                                        indices_B[idx_B].factor;
                    if (factor.is_zero()) continue;

                    auto data_A = A.data(0) + (indices_A[local_idx_A].offset + off_A_AB)*ts;

                    add(type, subcomm, cfg, group_A.dense_len, {}, group_AB.dense_len,
                        factor, conj_A, data_A, group_A.dense_stride[0],
                                                group_AB.dense_stride[0],
                           one,  false, data_B, {}, group_AB.dense_stride[1]);
                }
            });
        });
    });
}

void replicate_block(type_t type, const communicator& comm, const config& cfg,
                     const scalar& alpha, bool conj_A,
                     const indexed_marray_view<char>& A,
                     const dim_vector& idx_A_AB,
                     const indexed_marray_view<char>& B,
                     const dim_vector& idx_B_B,
                     const dim_vector& idx_B_AB)
{
    const len_type ts = type_size[type];

    index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    index_group<1> group_B(B, idx_B_B);

    group_indices<1> indices_A(type, A, group_AB, 0);
    group_indices<2> indices_B(type, B, group_AB, 1, group_B, 0);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    len_vector dense_len_B = group_AB.dense_len + group_B.dense_len;
    stride_vector dense_stride_B = group_AB.dense_stride[1] + group_B.dense_stride[0];

    scalar one(1.0, type);

    comm.do_tasks_deferred(nidx_B, stl_ext::prod(group_AB.dense_len)*
                                   stl_ext::prod(group_B.dense_len)*inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        for_each_match<false, true>(idx_A, nidx_A, indices_A, 0,
                                    idx_B, nidx_B, indices_B, 0,
        [&](stride_type next_B)
        {
            if (indices_A[idx_A].factor.is_zero()) return;

            for (auto local_idx_B = idx_B;local_idx_B < next_B;local_idx_B++)
            {
                auto factor = alpha*indices_A[idx_A].factor*
                                    indices_B[local_idx_B].factor;
                if (factor.is_zero()) continue;

                tasks.visit(idx++,
                [&,idx_A,local_idx_B,factor](const communicator& subcomm)
                {
                    stride_type off_A_AB, off_B_AB;
                    get_local_offset(indices_A[idx_A].idx[0], group_AB,
                                     off_A_AB, 0, off_B_AB, 1);

                    auto data_A = A.data(0) + (indices_A[idx_A].offset + off_A_AB)*ts;
                    auto data_B = B.data(0) + (indices_B[local_idx_B].offset + off_B_AB)*ts;

                    add(type, subcomm, cfg, {}, group_B.dense_len, group_AB.dense_len,
                        factor, conj_A, data_A, {}, group_AB.dense_stride[0],
                           one,  false, data_B, group_B.dense_stride[0],
                                                group_AB.dense_stride[1]);
                });
            }
        });
    });
}

void transpose_block(type_t type, const communicator& comm, const config& cfg,
                     const scalar& alpha, bool conj_A,
                     const indexed_marray_view<char>& A,
                     const dim_vector& idx_A_AB,
                     const indexed_marray_view<char>& B,
                     const dim_vector& idx_B_AB)
{
    const len_type ts = type_size[type];

    index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);

    group_indices<1> indices_A(type, A, group_AB, 0);
    group_indices<1> indices_B(type, B, group_AB, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    scalar one(1.0, type);

    comm.do_tasks_deferred(nidx_B, stl_ext::prod(group_AB.dense_len)*inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        for_each_match<false, false>(idx_A, nidx_A, indices_A, 0,
                                    idx_B, nidx_B, indices_B, 0,
        [&]
        {
            auto factor = alpha*indices_A[idx_A].factor*indices_B[idx_B].factor;
            if (factor.is_zero()) return;

            tasks.visit(idx++,
            [&,idx_A,idx_B,factor](const communicator& subcomm)
            {
                stride_type off_A_AB, off_B_AB;
                get_local_offset(indices_A[idx_A].idx[0], group_AB,
                                 off_A_AB, 0, off_B_AB, 1);

                auto data_A = A.data(0) + (indices_A[idx_A].offset + off_A_AB)*ts;
                auto data_B = B.data(0) + (indices_B[idx_B].offset + off_B_AB)*ts;

                add(type, subcomm, cfg, {}, {}, group_AB.dense_len,
                    factor, conj_A, data_A, {}, group_AB.dense_stride[0],
                       one,  false, data_B, {}, group_AB.dense_stride[1]);
            });
        });
    });
}

void add(type_t type, const communicator& comm, const config& cfg,
         const scalar& alpha, bool conj_A, const indexed_marray_view<char>& A,
         const dim_vector& idx_A_A,
         const dim_vector& idx_A_AB,
         const scalar&  beta, bool conj_B, const indexed_marray_view<char>& B,
         const dim_vector& idx_B_B,
         const dim_vector& idx_B_AB)
{
    if (beta.is_zero())
    {
        set(type, comm, cfg, beta, B, range(B.dimension()));
    }
    else if (!beta.is_one() || (beta.is_complex() && conj_B))
    {
        scale(type, comm, cfg, beta, conj_B, B, range(B.dimension()));
    }

    if (dpd_impl == FULL)
    {
        switch (type)
        {
            case TYPE_FLOAT:
                add_full(comm, cfg,
                         alpha.data.s, conj_A,
                         reinterpret_cast<const indexed_marray_view<float>&>(A), idx_A_A, idx_A_AB,
                         reinterpret_cast<const indexed_marray_view<float>&>(B), idx_B_B, idx_B_AB);
                break;
            case TYPE_DOUBLE:
                add_full(comm, cfg,
                         alpha.data.d, conj_A,
                         reinterpret_cast<const indexed_marray_view<double>&>(A), idx_A_A, idx_A_AB,
                         reinterpret_cast<const indexed_marray_view<double>&>(B), idx_B_B, idx_B_AB);
                break;
            case TYPE_SCOMPLEX:
                add_full(comm, cfg,
                         alpha.data.c, conj_A,
                         reinterpret_cast<const indexed_marray_view<scomplex>&>(A), idx_A_A, idx_A_AB,
                         reinterpret_cast<const indexed_marray_view<scomplex>&>(B), idx_B_B, idx_B_AB);
                break;
            case TYPE_DCOMPLEX:
                add_full(comm, cfg,
                         alpha.data.z, conj_A,
                         reinterpret_cast<const indexed_marray_view<dcomplex>&>(A), idx_A_A, idx_A_AB,
                         reinterpret_cast<const indexed_marray_view<dcomplex>&>(B), idx_B_B, idx_B_AB);
                break;
        }
    }
    else if (!idx_A_A.empty())
    {
        trace_block(type, comm, cfg,
                    alpha, conj_A, A, idx_A_A, idx_A_AB,
                                   B, idx_B_AB);
    }
    else if (!idx_B_B.empty())
    {
        replicate_block(type, comm, cfg,
                        alpha, conj_A, A, idx_A_AB,
                                       B, idx_B_B, idx_B_AB);
    }
    else
    {
        transpose_block(type, comm, cfg,
                        alpha, conj_A, A, idx_A_AB,
                                       B, idx_B_AB);
    }
}

}
}
