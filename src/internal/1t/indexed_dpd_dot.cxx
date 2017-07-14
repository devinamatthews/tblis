#include "indexed_dpd_dot.hpp"
#include "dot.hpp"
#include "internal/3t/indexed_dpd_mult.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dense_total_lengths_and_strides(const indexed_dpd_varray_view<T>& A, len_vector& len, stride_vector& stride)
{
    unsigned ndim = A.dense_dimension();
    unsigned nirrep = A.num_irreps();

    len.resize(ndim);
    stride.resize(ndim);

    for (unsigned i = 0;i < ndim;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            len[i] += A.length(i, irrep);
    }

    auto iperm = detail::inverse_permutation(A.permutation());
    stride[iperm[0]] = 1;
    for (unsigned i = 1;i < ndim;i++)
    {
        stride[iperm[i]] = stride[iperm[i-1]] * len[iperm[i-1]];
    }
}

template <typename T>
void indexed_dpd_dot_full(const communicator& comm, const config& cfg,
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
            detail::block_to_full(A, A2);
            detail::block_to_full(B, B2);
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
void indexed_dpd_dot_block(const communicator& comm, const config& cfg,
                           bool conj_A, const indexed_dpd_varray_view<const T>& A,
                           const dim_vector& idx_A_AB,
                           bool conj_B, const indexed_dpd_varray_view<const T>& B,
                           const dim_vector& idx_B_AB,
                           T& result)
{
    const unsigned nirrep = A.num_irreps();

    const unsigned ndim_A = A.dimension();
    const unsigned ndim_B = B.dimension();

    const unsigned dense_ndim_A = A.dense_dimension();
    const unsigned dense_ndim_B = B.dense_dimension();

    const unsigned batch_ndim_A = A.indexed_dimension();
    const unsigned batch_ndim_B = B.indexed_dimension();

    irrep_vector irreps_A(dense_ndim_A);
    irrep_vector irreps_B(dense_ndim_B);

    dim_vector dense_idx_A_AB;
    dim_vector dense_idx_B_AB;
    dim_vector mixed_idx_A_AB;
    dim_vector mixed_idx_B_AB;
    dim_vector mixed_pos_A_AB;
    dim_vector mixed_pos_B_AB;
    dim_vector batch_idx_A_AB;
    dim_vector batch_idx_B_AB;
    dim_vector batch_pos_A_AB;
    dim_vector batch_pos_B_AB;

    len_vector mixed_len_A_AB;
    len_vector mixed_len_B_AB;

    template <unsigned N>
    struct dpd_index_group
    {
        std::array<dim_vector,N> dense_idx;

        std::array<dim_vector,N> mixed_idx;
        std::array<dim_vector,N> mixed_pos;
        std::array<len_vector,N> mixed_len;

        std::array<dim_vector,N> batch_idx;
        std::array<dim_vector,N> batch_pos;

        template <typename... Args>
        dpd_index_group(const Args&... args)
        {
            //TODO
        }
    };

    unsigned dense_ndim_AB = 0;
    unsigned batch_ndim_AB = 0;
    for (unsigned i = 0;i < ndim_A;i++)
    {
        if (idx_A_AB[i] < dense_ndim_A && idx_B_AB[i] < dense_ndim_B)
        {
            dense_idx_A_AB.push_back(idx_A_AB[i]);
            dense_idx_B_AB.push_back(idx_B_AB[i]);
            dense_ndim_AB++;
        }
        else
        {
            if (idx_A_AB[i] < dense_ndim_A)
            {
                irreps_A[idx_A_AB[i]] = B.indexed_irrep(idx_B_AB[i]);
                mixed_idx_A_AB.push_back(idx_A_AB[i]);
                mixed_pos_A_AB.push_back(batch_ndim_AB);
                mixed_len_A_AB.push_back(A.length(idx_A_AB[i], irreps_A[idx_A_AB[i]]));
            }
            else
            {
                batch_idx_A_AB.push_back(idx_A_AB[i] - dense_ndim_A);
                batch_pos_A_AB.push_back(batch_ndim_AB);
            }

            if (idx_B_AB[i] < dense_ndim_B)
            {
                irreps_B[idx_B_AB[i]] = A.indexed_irrep(idx_A_AB[i]);
                mixed_idx_B_AB.push_back(idx_B_AB[i]);
                mixed_pos_B_AB.push_back(batch_ndim_AB);
                mixed_len_B_AB.push_back(B.length(idx_B_AB[i], irreps_B[idx_B_AB[i]]));
            }
            else
            {
                batch_idx_B_AB.push_back(idx_B_AB[i] - dense_ndim_B);
                batch_pos_B_AB.push_back(batch_ndim_AB);
            }

            batch_ndim_AB++;
        }
    }

    len_vector dense_len_A;
    len_vector dense_len_B;

    stride_vector dense_stride_A;
    stride_vector dense_stride_B;

    dense_total_lengths_and_strides(A, dense_len_A, dense_stride_A);
    dense_total_lengths_and_strides(B, dense_len_B, dense_stride_B);

    stride_type dense_AB = 1;
    stride_type nblock_AB = 1;
    for (unsigned i : dense_idx_A_AB)
    {
        dense_AB *= dense_len_A[i];
        nblock_AB *= nirrep;
    }
    dense_AB /= nblock_AB;

    if (nblock_AB > 1) nblock_AB /= nirrep;

    unsigned irrep_AB = A.irrep();

    auto dense_stride_A_AB = stl_ext::select_from(dense_stride_A, dense_idx_A_AB);
    auto dense_stride_B_AB = stl_ext::select_from(dense_stride_B, dense_idx_B_AB);

    auto reorder_AB = detail::sort_by_stride(dense_stride_A_AB, dense_stride_B_AB);

    stl_ext::permute(dense_idx_A_AB, reorder_AB);
    stl_ext::permute(dense_idx_B_AB, reorder_AB);

    stride_type nidx_A = A.num_indices()*stl_ext::prod(mixed_len_A_AB);
    stride_type nidx_B = B.num_indices()*stl_ext::prod(mixed_len_B_AB);

    std::vector<std::tuple<len_vector,len_type>> indices_A; indices_A.reserve(nidx_A);
    std::vector<std::tuple<len_vector,len_type>> indices_B; indices_B.reserve(nidx_B);

    viterator<0> iter_A(mixed_len_A_AB);
    for (len_type i = 0, j = 0;i < A.num_indices();i++)
    {
        len_vector idx(batch_ndim_AB);

        for (unsigned k = 0;k < batch_ndim_AB;k++)
            idx[batch_pos_A_AB[k]] = A.index(i, batch_idx_A_AB[k]);

        while (iter_A.next())
        {
            for (unsigned k = 0;k < iter_A.dimension();k++)
                idx[mixed_pos_A_AB[k]] = iter_A.position(k);

            indices_A.emplace_back(idx, i);
        }
    }

    viterator<0> iter_B(mixed_len_B_AB);
    for (len_type i = 0, j = 0;i < B.num_indices();i++)
    {
        len_vector idx(batch_ndim_AB);

        for (unsigned k = 0;k < batch_ndim_AB;k++)
            idx[batch_pos_B_AB[k]] = B.index(i, batch_idx_B_AB[k]);

        while (iter_B.next())
        {
            for (unsigned k = 0;k < iter_B.dimension();k++)
                idx[mixed_pos_B_AB[k]] = iter_B.position(k);

            indices_B.emplace_back(idx, i);
        }
    }

    stl_ext::sort(indices_A);
    stl_ext::sort(indices_B);

    auto dpd_A = A(0);
    auto dpd_B = B(0);

    dynamic_task_set tasks(comm, nidx_B*nblock_AB, stl_ext::prod(dense_AB));

    stride_type task = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    T local_result = T();
    T block_result;

    while (idx_A < nidx_A && idx_B < nidx_B)
    {
        if (get<0>(indices_A[idx_A]) < get<0>(indices_B[idx_B]))
        {
            idx_A++;
        }
        else if (get<0>(indices_A[idx_A]) > get<0>(indices_B[idx_B]))
        {
            idx_B++;
        }
        else
        {
            for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
            {
                tasks.visit(task++,
                [&,idx_A,idx_B,block_AB,irreps_A,irreps_B](const communicator& subcomm)
                {
                    detail::assign_irreps(dense_ndim_AB, irrep_AB, nirrep, block_AB,
                                          irreps_A, dense_idx_A_AB, irreps_B, dense_idx_B_AB);

                    if (detail::is_block_empty(dpd_A, irreps_A)) continue;
                    if (detail::is_block_empty(dpd_B, irreps_B)) continue;

                    auto local_A = dpd_A(irreps_A);
                    auto local_B = dpd_B(irreps_B);

                    auto len_AB = stl_ext::select_from(local_A.lengths(), dense_idx_A_AB);
                    auto stride_A_AB = stl_ext::select_from(local_A.strides(), dense_idx_A_AB);
                    auto stride_B_AB = stl_ext::select_from(local_B.strides(), dense_idx_B_AB);

                    auto data_A = local_A.data() + A.data(A.data(get<1>(indices_A[idx_A]))) - A.data(0);
                    auto data_B = local_B.data() + B.data(B.data(get<1>(indices_B[idx_B]))) - B.data(0);

                    for (unsigned i = 0;i < mixed_idx_A_AB.size();i++)
                    {
                        data_A += get<0>(indices_A[idx_A])[mixed_pos_A_AB[i]] *
                                  local_A.stride(mixed_idx_A_AB[i]);
                    }

                    for (unsigned i = 0;i < mixed_idx_B_AB.size();i++)
                    {
                        data_B += get<0>(indices_B[idx_B])[mixed_pos_B_AB[i]] *
                                  local_B.stride(mixed_idx_B_AB[i]);
                    }

                    dot(subcomm, cfg, len_AB,
                        conj_A, data_A, stride_A_AB,
                        conj_B, data_B, stride_B_AB,
                        block_result);

                    local_result += block_result;
                });
            }

            idx_A++;
            idx_B++;
        }
    }

    if (comm.master()) result = local_result;
}

template <typename T>
void indexed_dpd_dot(const communicator& comm, const config& cfg,
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
        indexed_dpd_dot_full(comm, cfg,
                             conj_A, A, idx_A_AB,
                             conj_B, B, idx_B_AB,
                             result);
    }
    else
    {
        indexed_dpd_dot_block(comm, cfg,
                              conj_A, A, idx_A_AB,
                              conj_B, B, idx_B_AB,
                              result);
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void indexed_dpd_dot(const communicator& comm, const config& cfg, \
                              bool conj_A, const indexed_dpd_varray_view<const T>& A, \
                              const dim_vector& idx_A_AB, \
                              bool conj_B, const indexed_dpd_varray_view<const T>& B, \
                              const dim_vector& idx_B_AB, \
                              T& result);
#include "configs/foreach_type.h"

}
}
