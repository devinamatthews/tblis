#ifndef _TBLIS_INTERNAL_1T_INDEXED_DPD_UTIL_HPP_
#define _TBLIS_INTERNAL_1T_INDEXED_DPD_UTIL_HPP_

#include "util/basic_types.h"
#include "internal/1t/indexed/util.hpp"
#include "internal/3t/dpd/mult.hpp"

namespace tblis
{
namespace internal
{

template <typename T, typename U>
void block_to_full(const indexed_dpd_varray_view<T>& A, varray<U>& A2)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim_A = A.dimension();
    unsigned dense_ndim_A = A.dense_dimension();
    unsigned idx_ndim_A = A.indexed_dimension();

    len_vector len_A(ndim_A);
    matrix<len_type> off_A({ndim_A, nirrep});
    for (unsigned i = 0;i < ndim_A;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            off_A[i][irrep] = len_A[i];
            len_A[i] += A.length(i, irrep);
        }
    }
    A2.reset(len_A);
    dim_vector split = range(1,dense_ndim_A);

    A.for_each_index(
    [&](const dpd_varray_view<T>& Ai, const len_vector& idx_A)
    {
        Ai.for_each_block(
        [&](const varray_view<T>& local_A, const irrep_vector& irreps_A)
        {
            varray_view<U> local_A2 = A2;

            for (unsigned i = 0;i < dense_ndim_A;i++)
            {
                local_A2.length(i, local_A.length(i));
                local_A2.shift(i, off_A[i][irreps_A[i]]);
            }
            for (unsigned i = dense_ndim_A;i < ndim_A;i++)
            {
                local_A2.shift(i, idx_A[i-dense_ndim_A] +
                    off_A[i][A.indexed_irrep(i-dense_ndim_A)]);
            }
            local_A2.lower(split);

            local_A2 = local_A;
        });
    });
}

template <typename T, typename U>
void full_to_block(const varray<U>& A2, const indexed_dpd_varray_view<T>& A)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim_A = A.dimension();
    unsigned dense_ndim_A = A.dense_dimension();
    unsigned idx_ndim_A = A.indexed_dimension();

    len_vector len_A(ndim_A);
    matrix<len_type> off_A({ndim_A, nirrep});
    for (unsigned i = 0;i < ndim_A;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            off_A[i][irrep] = len_A[i];
            len_A[i] += A.length(i, irrep);
        }
    }
    dim_vector split = range(1,dense_ndim_A);

    A.for_each_index(
    [&](const dpd_varray_view<T>& Ai, const len_vector& idx_A)
    {
        Ai.for_each_block(
        [&](const varray_view<T>& local_A, const irrep_vector& irreps_A)
        {
            varray_view<const U> local_A2 = A2;

            for (unsigned i = 0;i < dense_ndim_A;i++)
            {
                local_A2.length(i, local_A.length(i));
                local_A2.shift(i, off_A[i][irreps_A[i]]);
            }
            for (unsigned i = dense_ndim_A;i < ndim_A;i++)
            {
                local_A2.shift(i, idx_A[i-dense_ndim_A] +
                    off_A[i][A.indexed_irrep(i-dense_ndim_A)]);
            }
            local_A2.lower(split);

            local_A = local_A2;
        });
    });
}

template <unsigned I, size_t N>
void dense_total_lengths_and_strides_helper(std::array<len_vector,N>&,
                                            std::array<stride_vector,N>&) {}

template <unsigned I, size_t N, typename T, typename... Args>
void dense_total_lengths_and_strides_helper(std::array<len_vector,N>& len,
                                            std::array<stride_vector,N>& stride,
                                            const indexed_dpd_varray_view<T>& A,
                                            const dim_vector&, const Args&... args)
{
    unsigned ndim = A.dense_dimension();
    unsigned nirrep = A.num_irreps();

    len[I].resize(ndim);
    stride[I].resize(ndim);

    for (unsigned j = 0;j < ndim;j++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            len[I][j] += A.length(j, irrep);
    }

    auto iperm = detail::inverse_permutation(A.permutation());
    stride[I][iperm[0]] = 1;
    for (unsigned j = 1;j < ndim;j++)
    {
        stride[I][iperm[j]] = stride[I][iperm[j-1]] * len[I][iperm[j-1]];
    }

    dense_total_lengths_and_stride_helper<I+1>(len, stride, args...);
}

template <size_t N, typename... Args>
void dense_total_lengths_and_strides(std::array<len_vector,N>& len,
                                     std::array<stride_vector,N>& stride,
                                     const Args&... args)
{
    dense_total_lengths_and_stride_helper<0>(len, stride, args...);
}

template <unsigned N> struct dpd_index_group;

template <unsigned N>
void assign_dense_idx_helper(unsigned, unsigned, dpd_index_group<N>&) {}

template <unsigned N, typename T, typename... Args>
void assign_dense_idx_helper(unsigned i, unsigned j, dpd_index_group<N>& group,
                             const indexed_dpd_varray<T>& A,
                             const dim_vector& idx_A, const Args&... args)
{
    group.dense_idx[j].push_back(idx_A[i]);
    assign_dense_idx_helper(i, j+1, group, args...);
}

template <unsigned N, typename T, typename... Args>
void assign_dense_idx(unsigned i, dpd_index_group<N>& group,
                      const indexed_dpd_varray<T>& A,
                      const dim_vector& idx_A, const Args&... args)
{
    assign_dense_idx_helper(i, 0, group, A, idx_A, args...);
}

template <unsigned N>
void assign_mixed_or_batch_idx_helper(unsigned, unsigned, unsigned,
                                      dpd_index_group<N>&) {}

template <unsigned N, typename T, typename... Args>
void assign_mixed_or_batch_idx_helper(unsigned i, unsigned pos, unsigned j,
                                      dpd_index_group<N>& group,
                                      const indexed_dpd_varray_view<T>& A,
                                      const dim_vector& idx_A, const Args&... args)
{

    if (idx_A[i] < A.dense_dimension())
    {
        group.mixed_idx[j].push_back(idx_A[i]);
        group.mixed_pos[j].push_back(pos);
    }
    else
    {
        unsigned idx = idx_A[i] - A.dense_dimension();

        group.batch_idx[j].push_back(idx);
        group.batch_pos[j].push_back(pos);

        group.batch_irrep[pos] = A.indexed_irrep(idx);
        group.batch_len[pos] = A.indexed_length(idx);
    }

    assign_mixed_or_batch_idx_helper(i, pos, j+1, group, args...);
}

template <unsigned N, typename T, typename... Args>
void assign_mixed_or_batch_idx(unsigned i, unsigned pos,
                               dpd_index_group<N>& group,
                               const indexed_dpd_varray<T>& A,
                               const dim_vector& idx_A, const Args&... args)
{
    assign_mixed_or_batch_idx_helper(i, pos, 0, group,
                                     A, idx_A, args...);
}

template <unsigned N>
struct dpd_index_group
{
    unsigned dense_ndim = 0;
    unsigned batch_ndim = 0;
    unsigned dense_nblock = 1;
    stride_type dense_size = 0;

    std::array<dim_vector,N> dense_idx;

    std::array<dim_vector,N> mixed_idx;
    std::array<dim_vector,N> mixed_pos;

    len_vector batch_len;
    stride_vector batch_stride;
    irrep_vector batch_irrep;
    std::array<dim_vector,N> batch_idx;
    std::array<dim_vector,N> batch_pos;

    template <unsigned... I>
    dim_vector sort_by_stride(const std::array<stride_vector,N>& dense_stride,
                              detail::integer_sequence<unsigned, I...>)
    {
        return detail::sort_by_stride(dense_stride[I]...);
    }

    template <typename T, typename... Args>
    dpd_index_group(const indexed_dpd_varray<T>& A, const dim_vector& idx_A,
                    const Args&... args)
    {
        unsigned nirrep = A.num_irreps();

        batch_len.resize(idx_A.size());
        batch_irrep.resize(idx_A.size());

        for (unsigned i = 0;i < idx_A.size();i++)
        {
            if (is_idx_dense(i, A, idx_A, args...))
            {
                assign_dense_idx(i, *this, A, idx_A, args...);
                dense_ndim++;
            }
            else
            {
                assign_mixed_or_batch_idx(i, batch_ndim,
                                          *this, A, idx_A, args...);
                batch_ndim++;
            }
        }

        batch_len.resize(batch_ndim);
        batch_stride.resize(batch_ndim);
        batch_irrep.resize(batch_ndim);

        batch_stride[0] = 1;
        for (unsigned i = 1;i < batch_ndim;i++)
            batch_stride[i] = batch_stride[i-1]*batch_len[i-1];

        std::array<len_vector,N> dense_len;
        std::array<stride_vector,N> dense_stride;
        dense_total_lengths_and_strides(dense_len, dense_stride,
                                        A, idx_A, args...);

        dense_size = stl_ext::prod(batch_len);
        for (unsigned i = 0;i < dense_ndim;i++) dense_nblock *= nirrep;
        dense_size /= dense_nblock;

        if (dense_nblock > 1) dense_nblock /= nirrep;

        std::array<stride_vector,N> dense_stride_sub;
        for (unsigned i = 0;i < N;i++)
            dense_stride_sub[i] = stl_ext::select_from(dense_stride[i],
                                                       dense_idx[i]);

        auto reorder = sort_by_stride(dense_stride_sub,
                                      detail::static_range<unsigned, N>{});

        for (unsigned i = 0;i < N;i++)
            stl_ext::permute(dense_idx[i], reorder);
    }
};

template <unsigned N>
void assign_irreps_helper(unsigned i, const dpd_index_group<N>&) {}

template <unsigned N, typename... Args>
void assign_irreps_helper(unsigned i, const dpd_index_group<N>& group,
                          irrep_vector& irreps, Args&... args)
{
    for (unsigned j = 0;j < group.mixed_idx[i].size();j++)
    {
        irreps[group.mixed_idx[i][j]] = group.batch_irrep[group.mixed_pos[i][j]];
    }

    assign_irreps_helper(i+1, group, args...);
}

template <unsigned N, typename... Args>
void assign_irreps(const dpd_index_group<N>& group, Args&... args)
{
    assign_irreps_helper(0, group, args...);
}

template <unsigned I, unsigned N>
void get_local_geometry_helper(const len_vector& idx,
                               const dpd_index_group<N>& group,
                               len_vector& len) {}

template <unsigned I, unsigned N, typename T, typename... Args>
void get_local_geometry_helper(const len_vector& idx,
                               const dpd_index_group<N>& group,
                               len_vector& len,  const varray_view<T>& local_A,
                               stride_type& off, stride_type& stride,
                               unsigned i, const Args&... args)
{
    if (I == 0)
        len = stl_ext::select_from(local_A.lengths(), group.dense_idx[I]);

    stride = stl_ext::select_from(local_A.strides(), group.dense_idx[I]);

    off = 0;
    for (unsigned j = 0;j < group.mixed_idx[i].size();j++)
        off += idx[group.mixed_pos[i][j]]*local_A.stride(group.mixed_idx[i][j]);

    get_local_geometry_helper<I+1>(idx, group, len, args...);
}

template <unsigned N, typename... Args>
void get_local_geometry(const len_vector& idx, const dpd_index_group<N>& group,
                        len_vector& len, const Args&... args)
{
    get_local_geometry_helper<0>(idx, group, len, args...);
}

}
}

#endif
