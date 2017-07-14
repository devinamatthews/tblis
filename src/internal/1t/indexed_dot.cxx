#include "indexed_dot.hpp"
#include "dot.hpp"
#include "internal/3t/dpd_mult.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void indexed_dot_full(const communicator& comm, const config& cfg,
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
void indexed_dot_block(const communicator& comm, const config& cfg,
                       bool conj_A, const indexed_varray_view<const T>& A,
                       const dim_vector& idx_A_AB,
                       bool conj_B, const indexed_varray_view<const T>& B,
                       const dim_vector& idx_B_AB,
                       T& result)
{
    const unsigned ndim_A = A.dimension();
    const unsigned ndim_B = B.dimension();

    const unsigned dense_ndim_A = A.dense_dimension();
    const unsigned dense_ndim_B = B.dense_dimension();

    const unsigned batch_ndim_A = A.indexed_dimension();
    const unsigned batch_ndim_B = B.indexed_dimension();

    dim_vector mixed_idx_A_AB;
    dim_vector mixed_idx_B_AB;
    dim_vector batch_idx_A_AB(batch_ndim_A);
    dim_vector batch_idx_B_AB(batch_ndim_B);

    len_vector dense_len_AB;
    len_vector mixed_len_A_AB;
    len_vector mixed_len_B_AB;

    stride_vector dense_stride_A_AB;
    stride_vector dense_stride_B_AB;
    stride_vector mixed_stride_A_AB;
    stride_vector mixed_stride_B_AB;

    unsigned batch_ndim_AB = 0;
    for (unsigned i = 0;i < ndim_A;i++)
    {
        if (idx_A_AB[i] < dense_ndim_A)
        {
            dense_len_AB.push_back(A.dense_length(idx_A_AB[i]));
            dense_stride_A_AB.push_back(A.dense_stride(idx_A_AB[i]));
            dense_stride_B_AB.push_back(B.dense_stride(idx_A_AB[i]));
        }
        else
        {
            if (idx_A_AB[i] < dense_ndim_A)
            {
                mixed_idx_A_AB.push_back(batch_ndim_AB);
                mixed_len_A_AB.push_back(A.dense_length(idx_A_AB[i]));
                mixed_stride_A_AB.push_back(A.dense_stride(idx_A_AB[i]));
            }
            else
            {
                batch_idx_A_AB[idx_A_AB[i]-dense_ndim_A] = batch_ndim_AB;
            }

            if (idx_B_AB[i] < dense_ndim_B)
            {
                mixed_idx_B_AB.push_back(batch_ndim_AB);
                mixed_len_B_AB.push_back(B.dense_length(idx_B_AB[i]));
                mixed_stride_B_AB.push_back(B.dense_stride(idx_B_AB[i]));
            }
            else
            {
                batch_idx_B_AB[idx_B_AB[i]-dense_ndim_B] = batch_ndim_AB;
            }

            batch_ndim_AB++;
        }
    }

    label_vector dense_idx_AB; dense_idx_AB.resize(dense_len_AB.size());
    fold(dense_len_AB, dense_idx_AB, dense_stride_A_AB, dense_stride_B_AB);

    auto reorder_AB = detail::sort_by_stride(dense_stride_A_AB, dense_stride_B_AB);

    stl_ext::permute(dense_len_AB, reorder_AB);
    stl_ext::permute(dense_stride_A_AB, reorder_AB);
    stl_ext::permute(dense_stride_B_AB, reorder_AB);

    stride_type nidx_A_AB = A.num_indices()*stl_ext::prod(mixed_len_A_AB);
    stride_type nidx_B_AB = B.num_indices()*stl_ext::prod(mixed_len_B_AB);

    std::vector<std::pair<len_vector,T*>> indices_A; indices_A.reserve(nidx_A_AB);
    std::vector<std::pair<len_vector,T*>> indices_B; indices_B.reserve(nidx_B_AB);

    viterator<1> iter_A(mixed_len_A_AB, mixed_stride_A_AB);
    for (len_type i = 0, j = 0;i < A.num_indices();i++)
    {
        auto data = A.data(i);
        len_vector idx(batch_ndim_AB);

        for (unsigned k = 0;k < batch_ndim_A;k++)
            idx[batch_idx_A_AB[k]] = A.index(i,k);

        while (iter_A.next(data))
        {
            for (unsigned k = 0;k < iter_A.dimension();k++)
                idx[mixed_idx_A_AB[k]] = iter_A.position(k);

            indices_A.emplace_back(idx, data);
        }
    }

    viterator<1> iter_B(mixed_len_B_AB, mixed_stride_B_AB);
    for (len_type i = 0, j = 0;i < B.num_indices();i++)
    {
        auto data = B.data(i);
        len_vector idx(batch_ndim_AB);

        for (unsigned k = 0;k < batch_ndim_B;k++)
            idx[batch_idx_B_AB[k]] = B.index(i,k);

        while (iter_B.next(data))
        {
            for (unsigned k = 0;k < iter_B.dimension();k++)
                idx[mixed_idx_B_AB[k]] = iter_B.position(k);

            indices_B.emplace_back(idx, data);
        }
    }

    stl_ext::sort(indices_A);
    stl_ext::sort(indices_B);

    T local_result = T();
    T block_result;

    for (stride_type idx = 0, idx_A = 0, idx_B = 0;
         idx_A < nidx_A_AB && idx_B < nidx_B_AB;)
    {
        if (indices_A[idx_A].first < indices_B.first[idx_B])
        {
            idx_A++;
        }
        else if (indices_A[idx_A].first > indices_B.first[idx_B])
        {
            idx_B++
        }
        else
        {
            dot(comm, cfg, dense_len_AB,
                conj_A, indices_A[idx_A].second, dense_stride_A_AB,
                conj_B, indices_B[idx_B].second, dense_stride_B_AB,
                block_result);

            local_result += block_result;

            idx_A++;
            idx_B++;
        }
    }

    if (comm.master()) result = local_result;
}

template <typename T>
void indexed_dot(const communicator& comm, const config& cfg,
                 bool conj_A, const indexed_varray_view<const T>& A,
                 const dim_vector& idx_A_AB,
                 bool conj_B, const indexed_varray_view<const T>& B,
                 const dim_vector& idx_B_AB,
                 T& result)
{
    if (dpd_impl == FULL)
    {
        indexed_dot_full(comm, cfg,
                         conj_A, A, idx_A_AB,
                         conj_B, B, idx_B_AB,
                         result);
    }
    else
    {
        indexed_dot_block(comm, cfg,
                          conj_A, A, idx_A_AB,
                          conj_B, B, idx_B_AB,
                          result);
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void indexed_dot(const communicator& comm, const config& cfg, \
                          bool conj_A, const indexed_varray_view<const T>& A, \
                          const dim_vector& idx_A_AB, \
                          bool conj_B, const indexed_varray_view<const T>& B, \
                          const dim_vector& idx_B_AB, \
                          T& result);
#include "configs/foreach_type.h"

}
}
