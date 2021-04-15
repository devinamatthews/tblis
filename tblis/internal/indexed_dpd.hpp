#ifndef TBLIS_INTERNAL_INDEXED_DPD_HPP
#define TBLIS_INTERNAL_INDEXED_DPD_HPP

#include <tblis/internal/dpd.hpp>
#include <tblis/internal/indexed.hpp>

#include <marray/indexed_dpd_varray_view.hpp>

namespace tblis
{

using MArray::indexed_dpd_varray_view;

namespace internal
{

void add(type_t type, const communicator& comm, const config& cfg,
         const scalar& alpha, bool conj_A, const indexed_dpd_varray_view<char>& A,
         const dim_vector& idx_A,
         const dim_vector& idx_A_AB,
         const scalar&  beta, bool conj_B, const indexed_dpd_varray_view<char>& B,
         const dim_vector& idx_B,
         const dim_vector& idx_B_AB);

void dot(type_t type, const communicator& comm, const config& cfg,
         bool conj_A, const indexed_dpd_varray_view<char>& A,
         const dim_vector& idx_A_AB,
         bool conj_B, const indexed_dpd_varray_view<char>& B,
         const dim_vector& idx_B_AB,
         char* result);

void reduce(type_t type, const communicator& comm, const config& cfg, reduce_t op,
            const indexed_dpd_varray_view<char>& A, const dim_vector& idx_A_A,
            char* result, len_type& idx);

void scale(type_t type, const communicator& comm, const config& cfg,
           const scalar& alpha, bool conj_A, const indexed_dpd_varray_view<char>& A,
           const dim_vector& idx_A_A);

void set(type_t type, const communicator& comm, const config& cfg,
         const scalar& alpha, const indexed_dpd_varray_view<char>& A, const dim_vector& idx_A);

void shift(type_t type, const communicator& comm, const config& cfg,
           const scalar& alpha, const scalar& beta, bool conj_A,
           const indexed_dpd_varray_view<char>& A, const dim_vector& idx_A_A);

void mult(type_t type, const communicator& comm, const config& cfg,
          const scalar& alpha, bool conj_A, const indexed_dpd_varray_view<char>& A,
          const dim_vector& idx_A_AB,
          const dim_vector& idx_A_AC,
          const dim_vector& idx_A_ABC,
                               bool conj_B, const indexed_dpd_varray_view<char>& B,
          const dim_vector& idx_B_AB,
          const dim_vector& idx_B_BC,
          const dim_vector& idx_B_ABC,
          const scalar&  beta, bool conj_C, const indexed_dpd_varray_view<char>& C,
          const dim_vector& idx_C_AC,
          const dim_vector& idx_C_BC,
          const dim_vector& idx_C_ABC);

template <typename T>
void block_to_full(const communicator& comm, const config& cfg,
                   const indexed_dpd_varray_view<T>& A, varray<T>& A2)
{
    auto nirrep = A.num_irreps();
    auto ndim_A = A.dimension();
    auto dense_ndim_A = A.dense_dimension();

    len_vector len_A(ndim_A);
    matrix<len_type> off_A{{ndim_A, nirrep}};
    for (auto i : range(ndim_A))
    {
        for (auto irrep : range(nirrep))
        {
            off_A[i][irrep] = len_A[i];
            len_A[i] += A.length(i, irrep);
        }
    }

    if (comm.master()) A2.reset(len_A);
    comm.barrier();

    auto dense_stride_A2 = A2.strides();
    dense_stride_A2.resize(dense_ndim_A);

    A[0].for_each_block(
    [&](const varray_view<T>& local_A, const irrep_vector& irreps_A)
    {
        auto& dense_len_A = local_A.lengths();
        auto& dense_stride_A = local_A.strides();

        for (auto i : range(A.num_indices()))
        {
            auto data_A = local_A.data() + (A.data(i) - A.data(0));
            auto factor_A = A.factor(i);
            auto idx_A = A.indices(i);

            auto data_A2 = A2.data();
            for (auto i : range(dense_ndim_A))
                data_A2 += off_A[i][irreps_A[i]]*A2.stride(i);
            for (auto i : range(dense_ndim_A,ndim_A))
                data_A2 += (idx_A[i-dense_ndim_A] +
                    off_A[i][A.indexed_irrep(i-dense_ndim_A)])*A2.stride(i);

            add(type_tag<T>::value, comm, cfg, {}, {}, dense_len_A,
                factor_A, false, reinterpret_cast<char*>( data_A), {},  dense_stride_A,
                    T(0), false, reinterpret_cast<char*>(data_A2), {}, dense_stride_A2);
        }
    });
}

template <typename T>
void full_to_block(const communicator& comm, const config& cfg,
                   varray<T>& A2, const indexed_dpd_varray_view<T>& A)
{
    auto nirrep = A.num_irreps();
    auto ndim_A = A.dimension();
    auto dense_ndim_A = A.dense_dimension();

    matrix<len_type> off_A{{ndim_A, nirrep}};
    for (auto i : range(ndim_A))
    {
        len_type off = 0;
        for (auto irrep : range(nirrep))
        {
            off_A[i][irrep] = off;
            off += A.length(i, irrep);
        }
    }

    auto dense_stride_A2 = A2.strides();
    dense_stride_A2.resize(dense_ndim_A);

    A[0].for_each_block(
    [&](const varray_view<T>& local_A, const irrep_vector& irreps_A)
    {
        auto& dense_len_A = local_A.lengths();
        auto& dense_stride_A = local_A.strides();

        for (auto i : range(A.num_indices()))
        {
            auto data_A = local_A.data() + (A.data(i) - A.data(0));
            auto factor_A = A.factor(i);
            auto idx_A = A.indices(i);

            auto data_A2 = A2.data();
            for (auto i : range(dense_ndim_A))
                data_A2 += off_A[i][irreps_A[i]]*A2.stride(i);
            for (auto i : range(dense_ndim_A,ndim_A))
                data_A2 += (idx_A[i-dense_ndim_A] +
                    off_A[i][A.indexed_irrep(i-dense_ndim_A)])*A2.stride(i);

            add(type_tag<T>::value, comm, cfg, {}, {}, dense_len_A,
                factor_A, false, reinterpret_cast<char*>(data_A2), {}, dense_stride_A2,
                    T(1), false, reinterpret_cast<char*>( data_A), {},  dense_stride_A);
        }
    });
}

template <int N> struct dpd_index_group;

template <int I, int N>
void assign_dense_idx_helper(int, dpd_index_group<N>&) {}

template <int I, int N, typename T, typename... Args>
void assign_dense_idx_helper(int i, dpd_index_group<N>& group,
                             const indexed_dpd_varray_view<T>&,
                             const dim_vector& idx_A, const Args&... args)
{
    group.dense_idx[I].push_back(idx_A[i]);
    assign_dense_idx_helper<I+1>(i, group, args...);
}

template <int N, typename T, typename... Args>
void assign_dense_idx(int i, dpd_index_group<N>& group,
                      const indexed_dpd_varray_view<T>& A,
                      const dim_vector& idx_A, const Args&... args)
{
    assign_dense_idx_helper<0>(i, group, A, idx_A, args...);
}

template <int I, int N>
void assign_mixed_or_batch_idx_helper(int, int,
                                      dpd_index_group<N>&) {}

template <int I, int N, typename T, typename... Args>
void assign_mixed_or_batch_idx_helper(int i, int pos,
                                      dpd_index_group<N>& group,
                                      const indexed_dpd_varray_view<T>& A,
                                      const dim_vector& idx_A, const Args&... args)
{

    if (idx_A[i] < A.dense_dimension())
    {
        group.mixed_idx[I].push_back(idx_A[i]);
        group.mixed_pos[I].push_back(pos);
    }
    else
    {
        auto idx = idx_A[i] - A.dense_dimension();

        group.batch_idx[I].push_back(idx);
        group.batch_pos[I].push_back(pos);

        TBLIS_ASSERT(group.batch_irrep[pos] == -1 ||
                     group.batch_irrep[pos] == A.indexed_irrep(idx));
        TBLIS_ASSERT(group.batch_len[pos] == -1 ||
                     group.batch_len[pos] == A.indexed_length(idx));
        group.batch_irrep[pos] = A.indexed_irrep(idx);
        group.batch_len[pos] = A.indexed_length(idx);
    }

    assign_mixed_or_batch_idx_helper<I+1>(i, pos, group, args...);
}

template <int N, typename T, typename... Args>
void assign_mixed_or_batch_idx(int i, int pos,
                               dpd_index_group<N>& group,
                               const indexed_dpd_varray_view<T>& A,
                               const dim_vector& idx_A, const Args&... args)
{
    assign_mixed_or_batch_idx_helper<0>(i, pos, group,
                                        A, idx_A, args...);
}

template <int N>
struct dpd_index_group
{
    int dense_ndim = 0;
    int batch_ndim = 0;
    int dense_nblock = 1;
    stride_type dense_size = 0;
    bool pack_3d = false;

    std::array<dim_vector,N> dense_idx;

    std::array<dim_vector,N> mixed_idx;
    std::array<dim_vector,N> mixed_pos;

    len_vector batch_len;
    stride_vector batch_stride;
    irrep_vector batch_irrep;
    std::array<dim_vector,N> batch_idx;
    std::array<dim_vector,N> batch_pos;

    template <typename T, typename... Args>
    dpd_index_group(const indexed_dpd_varray_view<T>& A, const dim_vector& idx_A,
                    const Args&... args)
    {
        auto nirrep = A.num_irreps();

        batch_len.resize(idx_A.size(), -1);
        batch_irrep.resize(idx_A.size(), -1);

        for (auto i : range(idx_A.size()))
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

        if (batch_ndim > 0) batch_stride[0] = 1;
        for (auto i : range(1,batch_ndim))
            batch_stride[i] = batch_stride[i-1]*batch_len[i-1];

        std::array<len_vector,N> dense_len;
        std::array<stride_vector,N> dense_stride;
        dense_total_lengths_and_strides(dense_len, dense_stride,
                                        A, idx_A, args...);

        dense_size = 1;
        for (auto i : range(dense_ndim))
        {
            dense_size *= dense_len[0][i];
            dense_nblock *= nirrep;
        }

        if (dense_nblock > 1)
        {
            dense_size = std::max<stride_type>(1, dense_size/nirrep);
            dense_nblock /= nirrep;
        }

        auto unit = 0;
        for (auto i : range(N))
        {
            for (auto j : range(1,dense_ndim))
            {
                if (dense_stride[i][j] == 1)
                {
                    pack_3d = true;
                    unit = std::max(unit, j);
                    break;
                }
            }
        }

        if (pack_3d)
            for (auto i : range(N))
                std::rotate(dense_idx[i].begin()+1, dense_idx[i].begin()+unit, dense_idx[i].end());
    }
};

template <int I, int N>
void assign_irreps_helper(const dpd_index_group<N>&) {}

template <int I, int N, typename... Args>
void assign_irreps_helper(const dpd_index_group<N>& group,
                          irrep_vector& irreps, Args&... args)
{
    for (auto j : range(group.mixed_idx[I].size()))
    {
        irreps[group.mixed_idx[I][j]] = group.batch_irrep[group.mixed_pos[I][j]];
    }

    assign_irreps_helper<I+1>(group, args...);
}

template <int N, typename... Args>
void assign_irreps(const dpd_index_group<N>& group, Args&... args)
{
    assign_irreps_helper<0>(group, args...);
}

template <int I, int N>
void get_local_geometry_helper(const len_vector&,
                               const dpd_index_group<N>&,
                               len_vector&) {}

template <int I, int N, typename T, typename... Args>
void get_local_geometry_helper(const len_vector& idx,
                               const dpd_index_group<N>& group,
                               len_vector& len,  const varray_view<T>& local_A,
                               stride_vector& stride,
                               int, Args&&... args)
{
    if (I == 0)
        len = stl_ext::select_from(local_A.lengths(), group.dense_idx[I]);

    stride = stl_ext::select_from(local_A.strides(), group.dense_idx[I]);

    get_local_geometry_helper<I+1>(idx, group, len, std::forward<Args>(args)...);
}

template <int N, typename... Args>
void get_local_geometry(const len_vector& idx, const dpd_index_group<N>& group,
                        len_vector& len, Args&&... args)
{
    get_local_geometry_helper<0>(idx, group, len, std::forward<Args>(args)...);
}

template <int I, int N>
void get_local_offset_helper(const len_vector&,
                             const dpd_index_group<N>&) {}

template <int I, int N, typename T, typename... Args>
void get_local_offset_helper(const len_vector& idx,
                             const dpd_index_group<N>& group,
                             const T& A, stride_type& off,
                             int i, Args&&... args)
{
    off = 0;
    for (auto j : range(group.mixed_idx[i].size()))
        off += idx[group.mixed_pos[i][j]]*
            A.stride(group.mixed_idx[i][j]);

    get_local_offset_helper<I+1>(idx, group, std::forward<Args>(args)...);
}

template <int N, typename... Args>
void get_local_offset(const len_vector& idx, const dpd_index_group<N>& group,
                      Args&&... args)
{
    get_local_offset_helper<0>(idx, group, std::forward<Args>(args)...);
}

}
}

#endif //TBLIS_INTERNAL_INDEXED_DPD_HPP
