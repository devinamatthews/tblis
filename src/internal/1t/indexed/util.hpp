#ifndef _TBLIS_INTERNAL_1T_INDEXED_UTIL_HPP_
#define _TBLIS_INTERNAL_1T_INDEXED_UTIL_HPP_

#include "util/basic_types.h"
#include "util/tensor.hpp"
#include "internal/1t/dense/add.hpp"
#include "internal/3t/dpd/mult.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void block_to_full(const communicator& comm, const config& cfg,
                   const indexed_varray_view<T>& A, varray<T>& A2)
{
    unsigned ndim_A = A.dimension();
    unsigned dense_ndim_A = A.dense_dimension();

    if (comm.master()) A2.reset(A.lengths());
    comm.barrier();

    auto dense_len_A = A.dense_lengths();
    auto dense_stride_A = A.dense_strides();

    auto dense_stride_A2 = A2.strides();
    dense_stride_A2.resize(dense_ndim_A);

    for (len_type i = 0;i < A.num_indices();i++)
    {
        auto data_A = A.data(i);
        auto factor_A = A.factor(i);
        auto idx_A = A.indices(i);

        auto data_A2 = A2.data();
        for (unsigned i = dense_ndim_A;i < ndim_A;i++)
            data_A2 += idx_A[i-dense_ndim_A]*A2.stride(i);

        add(type_tag<T>::value, comm, cfg, {}, {}, dense_len_A,
            factor_A, false, reinterpret_cast<char*>( data_A), {},  dense_stride_A,
                T(0), false, reinterpret_cast<char*>(data_A2), {}, dense_stride_A2);
    }
}

template <typename T>
void full_to_block(const communicator& comm, const config& cfg,
                   varray<T>& A2, const indexed_varray_view<T>& A)
{
    unsigned ndim_A = A.dimension();
    unsigned dense_ndim_A = A.dense_dimension();

    auto dense_len_A = A.dense_lengths();
    auto dense_stride_A = A.dense_strides();

    auto dense_stride_A2 = A2.strides();
    dense_stride_A2.resize(dense_ndim_A);

    for (len_type i = 0;i < A.num_indices();i++)
    {
        auto data_A = A.data(i);
        auto factor_A = A.factor(i);
        auto idx_A = A.indices(i);

        auto data_A2 = A2.data();
        for (unsigned i = dense_ndim_A;i < ndim_A;i++)
            data_A2 += idx_A[i-dense_ndim_A]*A2.stride(i);

        add(type_tag<T>::value, comm, cfg, {}, {}, dense_len_A,
            factor_A, false, reinterpret_cast<char*>(data_A2), {}, dense_stride_A2,
                T(1), false, reinterpret_cast<char*>( data_A), {},  dense_stride_A);
    }
}

inline bool is_idx_dense(unsigned) { return true; }

template <typename Array, typename... Args>
bool is_idx_dense(unsigned i, const Array& A,
                  const dim_vector& idx_A, const Args&... args)
{
    return idx_A[i] < A.dense_dimension() && is_idx_dense(i, args...);
}

template <unsigned N> struct index_group;

template <unsigned I, unsigned N>
void assign_dense_idx_helper(unsigned, index_group<N>&) {}

template <unsigned I, unsigned N, typename T, typename... Args>
void assign_dense_idx_helper(unsigned i, index_group<N>& group,
                             const indexed_varray_view<T>& A,
                             const dim_vector& idx_A, const Args&... args)
{
    TBLIS_ASSERT(group.dense_len.back() == A.dense_length(idx_A[i]));
    group.dense_stride[I].push_back(A.dense_stride(idx_A[i]));
    assign_dense_idx_helper<I+1>(i, group, args...);
}

template <unsigned N, typename T, typename... Args>
void assign_dense_idx(unsigned i, index_group<N>& group,
                      const indexed_varray_view<T>& A,
                      const dim_vector& idx_A, const Args&... args)
{
    group.dense_len.push_back(A.dense_length(idx_A[i]));
    assign_dense_idx_helper<0>(i, group, A, idx_A, args...);
}

template <unsigned N>
void assign_mixed_or_batch_idx_helper(unsigned, unsigned, unsigned,
                                      index_group<N>&) {}

template <unsigned N, typename T, typename... Args>
void assign_mixed_or_batch_idx_helper(unsigned i, unsigned pos, unsigned j,
                                      index_group<N>& group,
                                      const indexed_varray_view<T>& A,
                                      const dim_vector& idx_A, const Args&... args)
{
    group.batch_len[pos] = A.length(idx_A[i]);

    if (idx_A[i] < A.dense_dimension())
    {
        group.mixed_stride[j].push_back(A.dense_stride(idx_A[i]));
        group.mixed_pos[j].push_back(pos);
    }
    else
    {
        unsigned idx = idx_A[i] - A.dense_dimension();

        group.batch_idx[j].push_back(idx);
        group.batch_pos[j].push_back(pos);
    }

    assign_mixed_or_batch_idx_helper(i, pos, j+1, group, args...);
}

template <unsigned N, typename T, typename... Args>
void assign_mixed_or_batch_idx(unsigned i, unsigned pos,
                               index_group<N>& group,
                               const indexed_varray_view<T>& A,
                               const dim_vector& idx_A, const Args&... args)
{
    assign_mixed_or_batch_idx_helper(i, pos, 0, group,
                                     A, idx_A, args...);
}

template <unsigned N>
struct index_group
{
    unsigned dense_ndim = 0;
    unsigned batch_ndim = 0;

    len_vector dense_len;
    std::array<stride_vector,N> dense_stride;

    std::array<stride_vector,N> mixed_stride;
    std::array<dim_vector,N> mixed_pos;

    len_vector batch_len;
    stride_vector batch_stride;
    std::array<dim_vector,N> batch_idx;
    std::array<dim_vector,N> batch_pos;

    template <unsigned... I>
    void fold(detail::integer_sequence<unsigned, I...>)
    {
        label_vector dense_idx; dense_idx.resize(dense_ndim);
        tblis::fold(dense_len, dense_idx, dense_stride[I]...);
    }

    template <unsigned... I>
    dim_vector sort_by_stride(detail::integer_sequence<unsigned, I...>)
    {
        return detail::sort_by_stride(dense_stride[I]...);
    }

    template <typename T, typename... Args>
    index_group(const indexed_varray_view<T>& A, const dim_vector& idx_A,
                const Args&... args)
    {
        batch_len.resize(idx_A.size());

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

        if (batch_ndim > 0) batch_stride[0] = 1;
        for (unsigned i = 1;i < batch_ndim;i++)
            batch_stride[i] = batch_stride[i-1]*batch_len[i-1];

        //fold(detail::static_range<unsigned, N>{});

        //auto reorder = sort_by_stride(detail::static_range<unsigned, N>{});

        //stl_ext::permute(dense_len, reorder);

        //for (unsigned i = 0;i < N;i++)
        //    stl_ext::permute(dense_stride[i], reorder);
    }
};

inline void get_mixed_lengths(len_vector&, dim_vector&) {}

template <typename Group, typename... Args>
void get_mixed_lengths(len_vector& len, dim_vector& off,
                       const Group& group, unsigned i, const Args&... args)
{
    off.push_back(len.size());

    for (unsigned j : group.mixed_pos[i])
    {
        len.push_back(group.batch_len[j]);
    }

    get_mixed_lengths(len, off, args...);
}

template <unsigned I, size_t N, typename Array>
void set_batch_indices_helper(std::array<len_vector,N>&,
                              std::array<stride_vector,N>&,
                              const Array&, unsigned) {}

template <unsigned I, size_t N, typename Array, typename Group,
          typename... Args>
void set_batch_indices_helper(std::array<len_vector,N>& idx,
                              std::array<stride_vector,N>& stride,
                              const Array& A, unsigned i,
                              const Group& group, unsigned j,
                              const Args&... args)
{
    idx[I].resize(group.batch_ndim);
    stride[I].resize(group.batch_ndim);

    for (unsigned k = 0;k < group.batch_idx[j].size();k++)
    {
        idx[I][group.batch_pos[j][k]] = A.index(i, group.batch_idx[j][k]);
        stride[I][group.batch_pos[j][k]] = group.batch_stride[group.batch_pos[j][k]];
    }

    set_batch_indices_helper<I+1>(idx, stride, A, i, args...);
}

template <size_t N, typename Array, typename... Args>
void set_batch_indices(std::array<len_vector,N>& idx,
                       std::array<stride_vector,N>& stride,
                       const Array& A, unsigned i, const Args&... args)
{
    set_batch_indices_helper<0>(idx, stride, A, i, args...);
}

template <size_t N>
void set_mixed_indices_helper(std::array<len_vector,N>&,
                              std::array<stride_vector,N>&,
                              const viterator<0>&, const dim_vector&) {}

template <unsigned I, size_t N, typename Group, typename... Args>
void set_mixed_indices_helper(std::array<len_vector,N>& idx,
                              std::array<stride_vector,N>& stride,
                              const viterator<0>& iter, const dim_vector& off,
                              const Group& group, unsigned j,
                              const Args&... args)
{
    for (unsigned k = 0;k < group.mixed_pos[j].size();k++)
    {
        idx[I][group.mixed_pos[j][k]] = iter.position()[off[I]+k];
        stride[I][group.mixed_pos[j][k]] = group.batch_stride[group.mixed_pos[j][k]];
    }

    set_mixed_indices_helper<I+1>(idx, stride, iter, off, args...);
}

template <size_t N, typename... Args>
void set_mixed_indices(std::array<len_vector,N>& idx,
                       std::array<stride_vector,N>& stride,
                       const viterator<0>& iter, const dim_vector& off,
                       const Args&... args)
{
    set_mixed_indices_helper<0>(idx, stride, iter, off, args...);
}

template <unsigned N>
struct index_set
{
    std::array<stride_type,N> key = {};
    std::array<len_vector,N> idx = {};
    stride_type offset = 0;
    scalar factor{0.0};
};

template <unsigned N>
struct group_indices : std::vector<index_set<N>>
{
    template <typename Array, typename... Args>
    group_indices(type_t type, const Array& A, const Args&... args)
    {
        const len_type ts = type_size[type];

        len_vector mixed_len;
        dim_vector mixed_off;
        get_mixed_lengths(mixed_len, mixed_off, args...);

        this->reserve(A.num_indices()*stl_ext::prod(mixed_len));

        viterator<0> iter(mixed_len);
        for (len_type i = 0;i < A.num_indices();i++)
        {
            index_set<N> idx;
            std::array<stride_vector,N> idx_stride;

            set_batch_indices(idx.idx, idx_stride, A, i, args...);

            idx.offset = (A.data(i) - A.data(0))/ts;

            switch (type)
            {
                case TYPE_FLOAT:    idx.factor.reset(reinterpret_cast<const    float*>(&A.factor(0))[i]); break;
                case TYPE_DOUBLE:   idx.factor.reset(reinterpret_cast<const   double*>(&A.factor(0))[i]); break;
                case TYPE_SCOMPLEX: idx.factor.reset(reinterpret_cast<const scomplex*>(&A.factor(0))[i]); break;
                case TYPE_DCOMPLEX: idx.factor.reset(reinterpret_cast<const dcomplex*>(&A.factor(0))[i]); break;
            }

            while (iter.next())
            {
                set_mixed_indices(idx.idx, idx_stride, iter, mixed_off, args...);

                for (unsigned j = 0;j < N;j++)
                {
                    idx.key[j] = 0;
                    for (unsigned k = 0;k < idx.idx[j].size();k++)
                    {
                        idx.key[j] += idx.idx[j][k]*idx_stride[j][k];
                    }
                }

                this->push_back(idx);
            }
        }

        stl_ext::sort(*this,
                      [](const index_set<N>& a, const index_set<N>& b)
                      {
                          return a.key < b.key;
                      });
    }
};

template <unsigned I, unsigned N>
void get_local_offset_helper(const len_vector&, const index_group<N>&) {}

template <unsigned I, unsigned N, typename... Args>
void get_local_offset_helper(const len_vector& idx, const index_group<N>& group,
                             stride_type& off, unsigned i, Args&&... args)
{
    off = 0;

    TBLIS_ASSERT(i < group.mixed_pos.size());
    TBLIS_ASSERT(i < group.mixed_stride.size());
    TBLIS_ASSERT(group.mixed_pos[i].size() ==
                 group.mixed_stride[i].size());

    for (unsigned j = 0;j < group.mixed_pos[i].size();j++)
    {
        TBLIS_ASSERT(group.mixed_pos[i][j] >= 0);
        TBLIS_ASSERT(group.mixed_pos[i][j] < idx.size());
        off += idx[group.mixed_pos[i][j]]*group.mixed_stride[i][j];
    }

    get_local_offset_helper<I+1>(idx, group, std::forward<Args>(args)...);
}

template <unsigned N, typename... Args>
void get_local_offset(const len_vector& idx, const index_group<N>& group,
                      Args&&... args)
{
    get_local_offset_helper<0>(idx, group, std::forward<Args>(args)...);
}

template <bool AIsRange, bool BIsRange, bool CIsRange> struct call_match_body;

template <> struct call_match_body<false, false, false>
{
    template <typename Body>
    call_match_body(Body&& body, stride_type, stride_type, stride_type)
    { body(); }
};

template <> struct call_match_body<true, false, false>
{
    template <typename Body>
    call_match_body(Body&& body, stride_type next_A, stride_type, stride_type)
    { body(next_A); }
};

template <> struct call_match_body<false, true, false>
{
    template <typename Body>
    call_match_body(Body&& body, stride_type, stride_type next_B, stride_type)
    { body(next_B); }
};

template <> struct call_match_body<true, true, false>
{
    template <typename Body>
    call_match_body(Body&& body, stride_type next_A, stride_type next_B, stride_type)
    { body(next_A, next_B); }
};

template <> struct call_match_body<false, false, true>
{
    template <typename Body>
    call_match_body(Body&& body, stride_type, stride_type, stride_type next_C)
    { body(next_C); }
};

template <> struct call_match_body<true, false, true>
{
    template <typename Body>
    call_match_body(Body&& body, stride_type next_A, stride_type, stride_type next_C)
    { body(next_A, next_C); }
};

template <> struct call_match_body<false, true, true>
{
    template <typename Body>
    call_match_body(Body&& body, stride_type, stride_type next_B, stride_type next_C)
    { body(next_B, next_C); }
};

template <> struct call_match_body<true, true, true>
{
    template <typename Body>
    call_match_body(Body&& body, stride_type next_A, stride_type next_B, stride_type next_C)
    { body(next_A, next_B, next_C); }
};

template <bool AIsRange, bool BIsRange, unsigned NA, unsigned NB,
          typename Body>
void for_each_match(stride_type& idx_A, stride_type nidx_A,
                   const group_indices<NA>& indices_A, unsigned iA,
                   stride_type& idx_B, stride_type nidx_B,
                   const group_indices<NB>& indices_B, unsigned iB,
                   Body&& body)
{
    while (idx_A < nidx_A && idx_B < nidx_B)
    {
        if (indices_A[idx_A].key[iA] < indices_B[idx_B].key[iB])
        {
            idx_A++;
            continue;
        }
        else if (indices_A[idx_A].key[iA] > indices_B[idx_B].key[iB])
        {
            idx_B++;
            continue;
        }

        auto next_A = idx_A+1;
        auto next_B = idx_B+1;

        if (AIsRange)
        {
            while (next_A < nidx_A &&
                   indices_A[next_A].key[iA] == indices_B[idx_B].key[iB])
                next_A++;
        }

        if (BIsRange)
        {
            while (next_B < nidx_B &&
                   indices_A[idx_A].key[iA] == indices_B[next_B].key[iB])
                next_B++;
        }

        call_match_body<AIsRange, BIsRange, false>{body, next_A, next_B, 0};

        idx_A = next_A;
        idx_B = next_B;
    }
}

template <bool AIsRange, bool BIsRange, bool CIsRange,
          unsigned NA, unsigned NB, unsigned NC, typename Body>
void for_each_match(stride_type& idx_A, stride_type nidx_A,
                   const group_indices<NA>& indices_A, unsigned iA,
                   stride_type& idx_B, stride_type nidx_B,
                   const group_indices<NB>& indices_B, unsigned iB,
                   stride_type& idx_C, stride_type nidx_C,
                   const group_indices<NC>& indices_C, unsigned iC,
                   Body&& body)
{
    while (idx_A < nidx_A && idx_B < nidx_B && idx_C < nidx_C)
    {
        if (indices_A[idx_A].key[iA] < indices_B[idx_B].key[iB])
        {
            if (indices_A[idx_A].key[iA] < indices_C[idx_C].key[iC])
            {
                idx_A++;
            }
            else if (indices_A[idx_A].key[iA] > indices_C[idx_C].key[iC])
            {
                idx_C++;
            }
            else
            {
                idx_A++;
                idx_C++;
            }
            continue;
        }
        else if (indices_A[idx_A].key[iA] > indices_B[idx_B].key[iB])
        {
            if (indices_B[idx_B].key[iB] < indices_C[idx_C].key[iC])
            {
                idx_B++;
            }
            else if (indices_B[idx_B].key[iB] > indices_C[idx_C].key[iC])
            {
                idx_C++;
            }
            else
            {
                idx_B++;
                idx_C++;
            }
            continue;
        }
        else if (indices_A[idx_A].key[iA] < indices_C[idx_C].key[iC])
        {
            idx_A++;
            idx_B++;
            continue;
        }
        else if (indices_A[idx_A].key[iA] > indices_C[idx_C].key[iC])
        {
            idx_C++;
            continue;
        }

        auto next_A = idx_A+1;
        auto next_B = idx_B+1;
        auto next_C = idx_C+1;

        if (AIsRange)
        {
            while (next_A < nidx_A &&
                   indices_A[next_A].key[iA] == indices_B[idx_B].key[iB])
                next_A++;
        }

        if (BIsRange)
        {
            while (next_B < nidx_B &&
                   indices_A[idx_A].key[iA] == indices_B[next_B].key[iB])
                next_B++;
        }

        if (CIsRange)
        {
            while (next_C < nidx_C &&
                   indices_A[idx_A].key[iA] == indices_C[next_C].key[iC])
                next_C++;
        }

        call_match_body<AIsRange, BIsRange, CIsRange>{body, next_A, next_B, next_C};

        idx_A = next_A;
        idx_B = next_B;
        idx_C = next_C;
    }
}

}
}

#endif
