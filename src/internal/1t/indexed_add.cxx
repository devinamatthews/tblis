#include "indexed_add.hpp"
#include "add.hpp"
#include "internal/3t/dpd_mult.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void indexed_add_full(const communicator& comm, const config& cfg,
                      T alpha, bool conj_A, const indexed_varray_view<const T>& A,
                      const dim_vector& idx_A_A,
                      const dim_vector& idx_A_AB,
                      T  beta, bool conj_B, const indexed_varray_view<      T>& B,
                      const dim_vector& idx_B_B,
                      const dim_vector& idx_B_AB)
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
            detail::full_to_block(B2, B);
        }
    },
    A2, B2);
}

template <typename T>
void indexed_trace_block(const communicator& comm, const config& cfg,
                         T alpha, bool conj_A, const indexed_varray_view<const T>& A,
                         const dim_vector& idx_A_A,
                         const dim_vector& idx_A_AB,
                         T  beta, bool conj_B, const indexed_varray_view<      T>& B,
                         const dim_vector& idx_B_AB)
{
    const unsigned ndim_A = A.dimension();
    const unsigned ndim_B = B.dimension();

    const unsigned dense_ndim_A = A.dense_dimension();
    const unsigned dense_ndim_B = B.dense_dimension();

    const unsigned batch_ndim_A = A.indexed_dimension();
    const unsigned batch_ndim_B = B.indexed_dimension();

    dim_vector mixed_pos_A_AB;
    dim_vector mixed_pos_B_AB;
    dim_vector batch_idx_A_A;
    dim_vector batch_idx_A_AB;
    dim_vector batch_idx_B_AB;
    dim_vector batch_pos_A_AB;
    dim_vector batch_pos_B_AB;

    len_vector dense_len_A;
    len_vector dense_len_AB;
    len_vector mixed_len_A_AB;
    len_vector mixed_len_B_AB;

    stride_vector dense_stride_A_A;
    stride_vector dense_stride_A_AB;
    stride_vector dense_stride_B_AB;
    stride_vector mixed_stride_A_AB;
    stride_vector mixed_stride_B_AB;

    unsigned batch_ndim_AB = 0;
    for (unsigned i = 0;i < ndim_A;i++)
    {
        if (idx_A_AB[i] < dense_ndim_A && idx_B_AB[i] < dense_ndim_B)
        {
            dense_len_AB.push_back(A.dense_length(idx_A_AB[i]));
            dense_stride_A_AB.push_back(A.dense_stride(idx_A_AB[i]));
            dense_stride_B_AB.push_back(B.dense_stride(idx_B_AB[i]));
        }
        else
        {
            if (idx_A_AB[i] < dense_ndim_A)
            {
                mixed_len_A_AB.push_back(A.dense_length(idx_A_AB[i]));
                mixed_stride_A_AB.push_back(A.dense_stride(idx_A_AB[i]));
                mixed_pos_A_AB.push_back(batch_ndim_AB);
            }
            else
            {
                batch_idx_A_AB.push_back(idx_A_AB[i] - dense_ndim_A);
                batch_pos_A_AB.push_back(batch_ndim_AB);
            }

            if (idx_B_AB[i] < dense_ndim_B)
            {
                mixed_len_B_AB.push_back(B.dense_length(idx_B_AB[i]));
                mixed_stride_B_AB.push_back(B.dense_stride(idx_B_AB[i]));
                mixed_pos_B_AB.push_back(batch_ndim_AB);
            }
            else
            {
                batch_idx_B_AB.push_back(idx_B_AB[i] - dense_ndim_B);
                batch_pos_B_AB.push_back(batch_ndim_AB);
            }

            batch_ndim_AB++;
        }
    }

    unsigned batch_ndim_A_only = 0;
    for (unsigned i : idx_A_A)
    {
        if (i < dense_ndim_A)
        {
            dense_len_A.push_back(A.dense_length(i));
            dense_stride_A_A.push_back(A.dense_stride(i));
        }
        else
        {
            batch_idx_A_A.push_back(i - dense_ndim_A);
            batch_ndim_A_only++;
        }
    }

    label_vector dense_idx_A; dense_idx_A.resize(dense_len_A.size());
    label_vector dense_idx_AB; dense_idx_AB.resize(dense_len_AB.size());
    fold(dense_len_A, dense_idx_A, dense_stride_A_A);
    fold(dense_len_AB, dense_idx_AB, dense_stride_A_AB, dense_stride_B_AB);

    auto reorder_AB = detail::sort_by_stride(dense_stride_A_AB, dense_stride_B_AB);

    stl_ext::permute(dense_len_AB, reorder_AB);
    stl_ext::permute(dense_stride_A_AB, reorder_AB);
    stl_ext::permute(dense_stride_B_AB, reorder_AB);

    stride_type nidx_A = A.num_indices()*stl_ext::prod(mixed_len_A_AB);
    stride_type nidx_B = B.num_indices()*stl_ext::prod(mixed_len_B_AB);

    std::vector<std::tuple<len_vector,len_vector,len_type>> indices_A; indices_A.reserve(nidx_A);
    std::vector<std::tuple<len_vector,len_type>> indices_B; indices_B.reserve(nidx_B);

    viterator<0> iter_A(mixed_len_A_AB);
    for (len_type i = 0;i < A.num_indices();i++)
    {
        len_vector idx_A(batch_ndim_A_only);
        len_vector idx_AB(batch_ndim_AB);

        for (unsigned k = 0;k < batch_ndim_A_only;k++)
            idx_A[k] = A.index(i, batch_idx_A_A[k]);

        for (unsigned k = 0;k < batch_ndim_AB;k++)
            idx_AB[batch_pos_A_AB[k]] = A.index(i, batch_idx_A_AB[k]);

        while (iter_A.next())
        {
            for (unsigned k = 0;k < iter_A.dimension();k++)
                idx_AB[mixed_pos_A_AB[k]] = iter_A.position(k);

            indices_A.emplace_back(idx_AB, idx_A, i);
        }
    }

    viterator<0> iter_B(mixed_len_B_AB);
    for (len_type i = 0;i < B.num_indices();i++)
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

    dynamic_task_set tasks(comm, nidx_B, stl_ext::prod(dense_len_AB));

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    while (idx_A < nidx_A && idx_B < nidx_B)
    {
        if (get<0>(indices_A[idx_A]) < get<0>(indices_B[idx_B]))
        {
            idx_A++;
        }
        else if (get<0>(indices_A[idx_A]) > get<0>(indices_B[idx_B]))
        {
            if (beta != T(1) || (is_complex<T>::value && conj_B))
            {
                tasks.visit(idx++,
                [&,idx_B](const communicator& subcomm)
                {
                    if (beta == T(0))
                    {
                        set(subcomm, cfg, B.dense_lengths(),
                            T(0), B.data(get<1>(indices_B[idx_B])), B.dense_strides());
                    }
                    else
                    {
                        scale(subcomm, cfg, B.dense_lengths(),
                              beta, conj_B, B.data(get<1>(indices_B[idx_B])), B.dense_strides());
                    }
                });
            }

            idx_B++;
        }
        else
        {
            stride_type next_A = idx_A;

            do { next_A++; }
            while (next_A < nidx_A && get<0>(indices_A[next_A]) == get<0>(indices_B[idx_B]));

            tasks.visit(idx++,
            [&,idx_A,idx_B,next_A,beta,conj_B](const communicator& subcomm)
            {
                auto data_B = B.data(get<1>(indices_B[idx_B]));

                stride_type off_A = 0;
                for (unsigned i = 0;i < mixed_pos_A_AB.size();i++)
                    off_A += get<0>(indices_A[idx_A])[mixed_pos_A_AB[i]]*mixed_stride_A_AB[i];

                if (!mixed_pos_B_AB.empty())
                {
                    //
                    // Pre-scale B since we will only by adding to a
                    // portion of it
                    //

                    if (beta == T(0))
                    {
                        set(subcomm, cfg, B.dense_lengths(),
                            T(0), data_B, B.dense_strides());
                    }
                    else if (beta != T(1) || (is_complex<T>::value && conj_B))
                    {
                        scale(subcomm, cfg, B.dense_lengths(),
                              beta, conj_B, data_B, B.dense_strides());
                    }

                    for (unsigned i = 0;i < mixed_pos_B_AB.size();i++)
                        data_B += get<0>(indices_B[idx_B])[mixed_pos_B_AB[i]]*mixed_stride_B_AB[i];

                    beta = T(1);
                    conj_B = false;
                }

                for (;idx_A < next_A;idx_A++)
                {
                    auto data_A = A.data(get<2>(indices_A[idx_A])) + off_A;

                    add(subcomm, cfg, dense_len_A, {}, dense_len_AB,
                        alpha, conj_A, data_A, dense_stride_A_A, dense_stride_A_AB,
                         beta, conj_B, {}, dense_stride_B_AB);

                    beta = T(1);
                    conj_B = false;
                }
            });

            idx_A = next_A;
            idx_B++;
        }
    }

    if (beta != T(1) || (is_complex<T>::value && conj_B))
    {
        while (idx_B < nidx_B)
        {
            tasks.visit(idx++,
            [&,idx_B](const communicator& subcomm)
            {
                if (beta == T(0))
                {
                    set(subcomm, cfg, B.dense_lengths(),
                        T(0), B.data(get<1>(indices_B[idx_B])), B.dense_strides());
                }
                else
                {
                    scale(subcomm, cfg, B.dense_lengths(),
                          beta, conj_B, B.data(get<1>(indices_B[idx_B])), B.dense_strides());
                }
            });

            idx_B++;
        }
    }
}

template <typename T>
void indexed_replicate_block(const communicator& comm, const config& cfg,
                             T alpha, bool conj_A, const indexed_varray_view<const T>& A,
                             const dim_vector& idx_A_AB,
                             T  beta, bool conj_B, const indexed_varray_view<      T>& B,
                             const dim_vector& idx_B_B,
                             const dim_vector& idx_B_AB)
{
    const unsigned ndim_A = A.dimension();
    const unsigned ndim_B = B.dimension();

    const unsigned dense_ndim_A = A.dense_dimension();
    const unsigned dense_ndim_B = B.dense_dimension();

    const unsigned batch_ndim_A = A.indexed_dimension();
    const unsigned batch_ndim_B = B.indexed_dimension();

    dim_vector mixed_pos_A_AB;
    dim_vector mixed_pos_B_AB;
    dim_vector batch_idx_B_B;
    dim_vector batch_idx_A_AB;
    dim_vector batch_idx_B_AB;
    dim_vector batch_pos_A_AB;
    dim_vector batch_pos_B_AB;

    len_vector dense_len_B;
    len_vector dense_len_AB;
    len_vector mixed_len_A_AB;
    len_vector mixed_len_B_AB;

    stride_vector dense_stride_B_B;
    stride_vector dense_stride_A_AB;
    stride_vector dense_stride_B_AB;
    stride_vector mixed_stride_A_AB;
    stride_vector mixed_stride_B_AB;

    unsigned batch_ndim_AB = 0;
    for (unsigned i = 0;i < ndim_A;i++)
    {
        if (idx_A_AB[i] < dense_ndim_A && idx_B_AB[i] < dense_ndim_B)
        {
            dense_len_AB.push_back(A.dense_length(idx_A_AB[i]));
            dense_stride_A_AB.push_back(A.dense_stride(idx_A_AB[i]));
            dense_stride_B_AB.push_back(B.dense_stride(idx_B_AB[i]));
        }
        else
        {
            if (idx_A_AB[i] < dense_ndim_A)
            {
                mixed_len_A_AB.push_back(A.dense_length(idx_A_AB[i]));
                mixed_stride_A_AB.push_back(A.dense_stride(idx_A_AB[i]));
                mixed_pos_A_AB.push_back(batch_ndim_AB);
            }
            else
            {
                batch_idx_A_AB.push_back(idx_A_AB[i] - dense_ndim_A);
                batch_pos_A_AB.push_back(batch_ndim_AB);
            }

            if (idx_B_AB[i] < dense_ndim_B)
            {
                mixed_len_B_AB.push_back(B.dense_length(idx_B_AB[i]));
                mixed_stride_B_AB.push_back(B.dense_stride(idx_B_AB[i]));
                mixed_pos_B_AB.push_back(batch_ndim_AB);
            }
            else
            {
                batch_idx_B_AB.push_back(idx_B_AB[i] - dense_ndim_B);
                batch_pos_B_AB.push_back(batch_ndim_AB);
            }

            batch_ndim_AB++;
        }
    }

    unsigned batch_ndim_B_only = 0;
    for (unsigned i : idx_B_B)
    {
        if (i < dense_ndim_B)
        {
            dense_len_B.push_back(B.dense_length(i));
            dense_stride_B_B.push_back(B.dense_stride(i));
        }
        else
        {
            batch_idx_B_B.push_back(i - dense_ndim_B);
            batch_ndim_B_only++;
        }
    }

    label_vector dense_idx_B; dense_idx_B.resize(dense_len_B.size());
    label_vector dense_idx_AB; dense_idx_AB.resize(dense_len_AB.size());
    fold(dense_len_B, dense_idx_B, dense_stride_B_B);
    fold(dense_len_AB, dense_idx_AB, dense_stride_A_AB, dense_stride_B_AB);

    auto reorder_AB = detail::sort_by_stride(dense_stride_A_AB, dense_stride_B_AB);

    stl_ext::permute(dense_len_AB, reorder_AB);
    stl_ext::permute(dense_stride_A_AB, reorder_AB);
    stl_ext::permute(dense_stride_B_AB, reorder_AB);

    stride_type nidx_A = A.num_indices()*stl_ext::prod(mixed_len_A_AB);
    stride_type nidx_B = B.num_indices()*stl_ext::prod(mixed_len_B_AB);

    std::vector<std::tuple<len_vector,len_type>> indices_A; indices_A.reserve(nidx_A);
    std::vector<std::tuple<len_vector,len_vector,len_type>> indices_B; indices_B.reserve(nidx_B);

    viterator<0> iter_A(mixed_len_A_AB);
    for (len_type i = 0;i < A.num_indices();i++)
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
    for (len_type i = 0;i < B.num_indices();i++)
    {
        len_vector idx_B(batch_ndim_B_only);
        len_vector idx_AB(batch_ndim_AB);

        for (unsigned k = 0;k < batch_ndim_B_only;k++)
            idx_B[k] = B.index(i, batch_idx_B_B[k]);

        for (unsigned k = 0;k < batch_ndim_AB;k++)
            idx_AB[batch_pos_B_AB[k]] = B.index(i, batch_idx_B_AB[k]);

        while (iter_B.next())
        {
            for (unsigned k = 0;k < iter_B.dimension();k++)
                idx_AB[mixed_pos_B_AB[k]] = iter_B.position(k);

            indices_B.emplace_back(idx_AB, idx_B, i);
        }
    }

    stl_ext::sort(indices_A);
    stl_ext::sort(indices_B);

    dynamic_task_set tasks(comm, nidx_B, stl_ext::prod(dense_len_AB)*stl_ext::prod(dense_len_B));

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    while (idx_A < nidx_A && idx_B < nidx_B)
    {
        if (get<0>(indices_A[idx_A]) < get<0>(indices_B[idx_B]))
        {
            idx_A++;
        }
        else if (get<0>(indices_A[idx_A]) > get<0>(indices_B[idx_B]))
        {
            if (beta != T(1) || (is_complex<T>::value && conj_B))
            {
                tasks.visit(idx++,
                [&,idx_B](const communicator& subcomm)
                {
                    if (beta == T(0))
                    {
                        set(subcomm, cfg, B.dense_lengths(),
                            T(0), B.data(get<2>(indices_B[idx_B])), B.dense_strides());
                    }
                    else
                    {
                        scale(subcomm, cfg, B.dense_lengths(),
                              beta, conj_B, B.data(get<2>(indices_B[idx_B])), B.dense_strides());
                    }
                });
            }

            idx_B++;
        }
        else
        {
            do
            {
                tasks.visit(idx++,
                [&,idx_A,idx_B](const communicator& subcomm)
                {
                    auto data_A = A.data(get<1>(indices_A[idx_A]));
                    auto data_B = B.data(get<2>(indices_B[idx_B]));

                    for (unsigned i = 0;i < mixed_pos_A_AB.size();i++)
                        data_A += get<0>(indices_A[idx_A])[mixed_pos_A_AB[i]]*mixed_stride_A_AB[i];

                    if (!mixed_pos_B_AB.empty())
                    {
                        //
                        // Pre-scale B since we will only by adding to a
                        // portion of it
                        //

                        if (beta == T(0))
                        {
                            set(subcomm, cfg, B.dense_lengths(),
                                T(0), data_B, B.dense_strides());
                        }
                        else if (beta != T(1) || (is_complex<T>::value && conj_B))
                        {
                            scale(subcomm, cfg, B.dense_lengths(),
                                  beta, conj_B, data_B, B.dense_strides());
                        }

                        for (unsigned i = 0;i < mixed_pos_B_AB.size();i++)
                            data_B += get<0>(indices_B[idx_B])[mixed_pos_B_AB[i]]*mixed_stride_B_AB[i];

                        add(subcomm, cfg, {}, dense_len_B, dense_len_AB,
                            alpha, conj_A, data_A, {}, dense_stride_A_AB,
                             T(1),  false, data_B, dense_stride_B_B, dense_stride_B_AB);
                    }
                    else
                    {
                        add(subcomm, cfg, {}, dense_len_B, dense_len_AB,
                            alpha, conj_A, data_A, {}, dense_stride_A_AB,
                             beta, conj_B, data_B, dense_stride_B_B, dense_stride_B_AB);
                    }
                });

                idx_B++;
            }
            while (idx_B < nidx_B && get<0>(indices_A[idx_A]) == get<0>(indices_B[idx_B]));

            idx_A++;
        }
    }

    if (beta != T(1) || (is_complex<T>::value && conj_B))
    {
        while (idx_B < nidx_B)
        {
            tasks.visit(idx++,
            [&,idx_B](const communicator& subcomm)
            {
                if (beta == T(0))
                {
                    set(subcomm, cfg, B.dense_lengths(),
                        T(0), B.data(get<2>(indices_B[idx_B])), B.dense_strides());
                }
                else
                {
                    scale(subcomm, cfg, B.dense_lengths(),
                          beta, conj_B, B.data(get<2>(indices_B[idx_B])), B.dense_strides());
                }
            });

            idx_B++;
        }
    }
}

template <typename T>
void indexed_transpose_block(const communicator& comm, const config& cfg,
                             T alpha, bool conj_A, const indexed_varray_view<const T>& A,
                             const dim_vector& idx_A_AB,
                             T  beta, bool conj_B, const indexed_varray_view<      T>& B,
                             const dim_vector& idx_B_AB)
{
    const unsigned ndim_A = A.dimension();
    const unsigned ndim_B = B.dimension();

    const unsigned dense_ndim_A = A.dense_dimension();
    const unsigned dense_ndim_B = B.dense_dimension();

    const unsigned batch_ndim_A = A.indexed_dimension();
    const unsigned batch_ndim_B = B.indexed_dimension();

    dim_vector mixed_pos_A_AB;
    dim_vector mixed_pos_B_AB;
    dim_vector batch_idx_A_AB;
    dim_vector batch_idx_B_AB;
    dim_vector batch_pos_A_AB;
    dim_vector batch_pos_B_AB;

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
        if (idx_A_AB[i] < dense_ndim_A && idx_B_AB[i] < dense_ndim_B)
        {
            dense_len_AB.push_back(A.dense_length(idx_A_AB[i]));
            dense_stride_A_AB.push_back(A.dense_stride(idx_A_AB[i]));
            dense_stride_B_AB.push_back(B.dense_stride(idx_B_AB[i]));
        }
        else
        {
            if (idx_A_AB[i] < dense_ndim_A)
            {
                mixed_len_A_AB.push_back(A.dense_length(idx_A_AB[i]));
                mixed_stride_A_AB.push_back(A.dense_stride(idx_A_AB[i]));
                mixed_pos_A_AB.push_back(batch_ndim_AB);
            }
            else
            {
                batch_idx_A_AB.push_back(idx_A_AB[i] - dense_ndim_A);
                batch_pos_A_AB.push_back(batch_ndim_AB);
            }

            if (idx_B_AB[i] < dense_ndim_B)
            {
                mixed_len_B_AB.push_back(B.dense_length(idx_B_AB[i]));
                mixed_stride_B_AB.push_back(B.dense_stride(idx_B_AB[i]));
                mixed_pos_B_AB.push_back(batch_ndim_AB);
            }
            else
            {
                batch_idx_B_AB.push_back(idx_B_AB[i] - dense_ndim_B);
                batch_pos_B_AB.push_back(batch_ndim_AB);
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

    stride_type nidx_A = A.num_indices()*stl_ext::prod(mixed_len_A_AB);
    stride_type nidx_B = B.num_indices()*stl_ext::prod(mixed_len_B_AB);

    std::vector<std::tuple<len_vector,len_type>> indices_A; indices_A.reserve(nidx_A);
    std::vector<std::tuple<len_vector,len_type>> indices_B; indices_B.reserve(nidx_B);

    viterator<0> iter_A(mixed_len_A_AB);
    for (len_type i = 0;i < A.num_indices();i++)
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
    for (len_type i = 0;i < B.num_indices();i++)
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

    dynamic_task_set tasks(comm, nidx_B, stl_ext::prod(dense_len_AB));

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    while (idx_A < nidx_A && idx_B < nidx_B)
    {
        if (get<0>(indices_A[idx_A]) < get<0>(indices_B[idx_B]))
        {
            idx_A++;
        }
        else if (get<0>(indices_A[idx_A]) > get<0>(indices_B[idx_B]))
        {
            if (beta != T(1) || (is_complex<T>::value && conj_B))
            {
                tasks.visit(idx++,
                [&,idx_B](const communicator& subcomm)
                {
                    if (beta == T(0))
                    {
                        set(subcomm, cfg, B.dense_lengths(),
                            T(0), B.data(get<1>(indices_B[idx_B])), B.dense_strides());
                    }
                    else
                    {
                        scale(subcomm, cfg, B.dense_lengths(),
                              beta, conj_B, B.data(get<1>(indices_B[idx_B])), B.dense_strides());
                    }
                });
            }

            idx_B++
        }
        else
        {
            tasks.visit(idx++,
            [&,idx_A,idx_B](const communicator& subcomm)
            {
                auto data_A = A.data(get<1>(indices_A[idx_A]));
                auto data_B = B.data(get<1>(indices_B[idx_B]));

                for (unsigned i = 0;i < mixed_pos_A_AB.size();i++)
                    data_A += get<0>(indices_A[idx_A])[mixed_pos_A_AB[i]]*mixed_stride_A_AB[i];

                if (!mixed_pos_B_AB.empty())
                {
                    //
                    // Pre-scale B since we will only by adding to a
                    // portion of it
                    //

                    if (beta == T(0))
                    {
                        set(subcomm, cfg, B.dense_lengths(),
                            T(0), data_B, B.dense_strides());
                    }
                    else if (beta != T(1) || (is_complex<T>::value && conj_B))
                    {
                        scale(subcomm, cfg, B.dense_lengths(),
                              beta, conj_B, data_B, B.dense_strides());
                    }

                    for (unsigned i = 0;i < mixed_pos_B_AB.size();i++)
                        data_B += get<0>(indices_B[idx_B])[mixed_pos_B_AB[i]]*mixed_stride_B_AB[i];

                    add(subcomm, cfg, {}, {}, dense_len_AB,
                        alpha, conj_A, data_A, {}, dense_stride_A_AB,
                         T(1),  false, data_B, {}, dense_stride_B_AB);
                }
                else
                {
                    add(subcomm, cfg, {}, {}, dense_len_AB,
                        alpha, conj_A, data_A, {}, dense_stride_A_AB,
                         beta, conj_B, data_B, {}, dense_stride_B_AB);
                }
            });

            idx_A++;
            idx_B++;
        }
    }

    if (beta != T(1) || (is_complex<T>::value && conj_B))
    {
        while (idx_B < nidx_B)
        {
            tasks.visit(idx++,
            [&,idx_B](const communicator& subcomm)
            {
                if (beta == T(0))
                {
                    set(subcomm, cfg, B.dense_lengths(),
                        T(0), B.data(get<1>(indices_B[idx_B])), B.dense_strides());
                }
                else
                {
                    scale(subcomm, cfg, B.dense_lengths(),
                          beta, conj_B, B.data(get<1>(indices_B[idx_B])), B.dense_strides());
                }
            });

            idx_B++;
        }
    }
}

template <typename T>
void indexed_add(const communicator& comm, const config& cfg,
                 T alpha, bool conj_A, const indexed_varray_view<const T>& A,
                 const dim_vector& idx_A_A,
                 const dim_vector& idx_A_AB,
                 T  beta, bool conj_B, const indexed_varray_view<      T>& B,
                 const dim_vector& idx_B_B,
                 const dim_vector& idx_B_AB)
{
    if (dpd_impl == FULL)
    {
        indexed_add_full(comm, cfg,
                         alpha, conj_A, A, idx_A_A, idx_A_AB,
                          beta, conj_B, B, idx_B_B, idx_B_AB);
    }
    else if (!idx_A_A.empty())
    {
        indexed_trace_block(comm, cfg,
                            alpha, conj_A, A, idx_A_A, idx_A_AB,
                             beta, conj_B, B, idx_B_AB);
    }
    else if (!idx_B_B.empty())
    {
        indexed_replicate_block(comm, cfg,
                                alpha, conj_A, A, idx_A_AB,
                                 beta, conj_B, B, idx_B_B, idx_B_AB);
    }
    else
    {
        indexed_transpose_block(comm, cfg,
                                alpha, conj_A, A, idx_A_AB,
                                 beta, conj_B, B, idx_B_AB);
    }
}

#define FOREACH_TYPE(T) \
template void indexed_add(const communicator& comm, const config& cfg, \
                          T alpha, bool conj_A, const indexed_varray_view<const T>& A, \
                          const dim_vector& idx_A, \
                          const dim_vector& idx_A_AB, \
                          T  beta, bool conj_B, const indexed_varray_view<      T>& B, \
                          const dim_vector& idx_B, \
                          const dim_vector& idx_B_AB);
#include "configs/foreach_type.h"

}
}
