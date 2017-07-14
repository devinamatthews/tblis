#include "indexed_dpd_add.hpp"
#include "add.hpp"
#include "internal/3t/dpd_mult.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void indexed_dpd_add_full(const communicator& comm, const config& cfg,
                          T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A,
                          const dim_vector& idx_A_A,
                          const dim_vector& idx_A_AB,
                          T  beta, bool conj_B, const indexed_dpd_varray_view<      T>& B,
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
void indexed_dpd_trace_block(const communicator& comm, const config& cfg,
                             T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A,
                             const dim_vector& idx_A_A,
                             const dim_vector& idx_A_AB,
                             T  beta, bool conj_B, const indexed_dpd_varray_view<      T>& B,
                             const dim_vector& idx_B_AB)
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

    dim_vector dense_idx_A_A;
    dim_vector dense_idx_A_AB;
    dim_vector dense_idx_B_AB;
    dim_vector mixed_idx_A_AB;
    dim_vector mixed_idx_B_AB;
    dim_vector mixed_pos_A_AB;
    dim_vector mixed_pos_B_AB;
    dim_vector batch_idx_A_A;
    dim_vector batch_idx_A_AB;
    dim_vector batch_idx_B_AB;
    dim_vector batch_pos_A_AB;
    dim_vector batch_pos_B_AB;

    len_vector mixed_len_A_AB;
    len_vector mixed_len_B_AB;

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

    unsigned dense_ndim_A_only = 0;
    unsigned batch_ndim_A_only = 0;
    for (unsigned i : idx_A_A)
    {
        if (i < dense_ndim_B)
        {
            dense_idx_A_A.push_back(i);
            dense_ndim_A_only++;
        }
        else
        {
            batch_idx_A_A.push_back(i - dense_ndim_A);
            batch_ndim_A_only++;
        }
    }

    len_vector dense_len_A(ndim_A);
    for (unsigned i = 0;i < dense_ndim_A;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            dense_len_A[i] += A.length(i, irrep);
    }

    len_vector dense_len_B(ndim_B);
    for (unsigned i = 0;i < dense_ndim_B;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            dense_len_B[i] += B.length(i, irrep);
    }

    auto iperm_A = detail::inverse_permutation(A.permutation());
    stride_vector dense_stride_A(dense_ndim_A, 1);
    for (unsigned i = 1;i < dense_ndim_A;i++)
    {
        dense_stride_A[iperm_A[i]] = dense_stride_A[iperm_A[i-1]] *
                                     dense_len_A[iperm_A[i-1]];
    }

    auto iperm_B = detail::inverse_permutation(B.permutation());
    stride_vector dense_stride_B(dense_ndim_B, 1);
    for (unsigned i = 1;i < dense_ndim_B;i++)
    {
        dense_stride_B[iperm_B[i]] = dense_stride_B[iperm_B[i-1]] *
                                     dense_len_B[iperm_B[i-1]];
    }

    stride_type dense_A = 1;
    stride_type nblock_A = 1;
    for (unsigned i : dense_idx_A_A)
    {
        dense_A *= dense_len_A[i];
        nblock_A *= nirrep;
    }
    dense_A /= nblock_A;

    stride_type dense_AB = 1;
    stride_type nblock_AB = 1;
    for (unsigned i : dense_idx_A_AB)
    {
        dense_AB *= dense_len_A[i];
        nblock_AB *= nirrep;
    }
    dense_AB /= nblock_AB;

    if (nblock_A > 1) nblock_A /= nirrep;
    if (nblock_AB > 1) nblock_AB /= nirrep;

    unsigned irrep_AB = B.irrep();
    unsigned irrep_A = B.irrep()^A.irrep();

    auto dense_stride_A_A = stl_ext::select_from(dense_stride_A, dense_idx_A_A);
    auto dense_stride_A_AB = stl_ext::select_from(dense_stride_A, dense_idx_A_AB);
    auto dense_stride_B_AB = stl_ext::select_from(dense_stride_B, dense_idx_B_AB);

    auto reorder_A = detail::sort_by_stride(dense_stride_A_A);
    auto reorder_AB = detail::sort_by_stride(dense_stride_A_AB, dense_stride_B_AB);

    stl_ext::permute(dense_idx_A_A, reorder_A);
    stl_ext::permute(dense_idx_A_AB, reorder_AB);
    stl_ext::permute(dense_idx_B_AB, reorder_AB);

    stride_type nidx_A = A.num_indices()*stl_ext::prod(mixed_len_A_AB);
    stride_type nidx_B = B.num_indices()*stl_ext::prod(mixed_len_B_AB);

    std::vector<std::tuple<len_vector,len_vector,len_type>> indices_A; indices_A.reserve(nidx_A);
    std::vector<std::tuple<len_vector,len_type>> indices_B; indices_B.reserve(nidx_B);

    viterator<0> iter_A(mixed_len_A_AB);
    for (len_type i = 0;i < A.num_indices();i++)
    {
        len_vector idx_AB(batch_ndim_AB);
        len_vector idx_A(batch_ndim_A_only);

        for (unsigned k = 0;k < batch_ndim_AB;k++)
            idx_AB[batch_pos_A_AB[k]] = A.index(i, batch_idx_A_AB[k]);

        for (unsigned k = 0;k < batch_ndim_A_only;k++)
            idx_A[k] = B.index(i, batch_idx_A_A[k]);

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

    auto dpd_A = A(0);
    auto dpd_B = B(0);

    dynamic_task_set tasks(comm, nidx_B*nblock_AB, dense_AB);

    stride_type task = 0;
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
                for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
                {
                    tasks.visit(task++,
                    [&,idx_B,block_AB,irreps_B](const communicator& subcomm)
                    {
                        detail::assign_irreps(dense_ndim_AB, irrep_AB, nirrep, block_AB,
                                              irreps_B, dense_idx_B_AB);

                        if (detail::is_block_empty(dpd_B, irreps_B)) continue;

                        auto local_B = dpd_B(irreps_B);

                        auto data_B = local_B.data() + B.data(B.data(get<1>(indices_B[idx_B]))) - B.data(0);

                        if (beta == T(0))
                        {
                            set(subcomm, cfg, local_B.lengths(),
                                T(0), data_B, local_B.strides());
                        }
                        else
                        {
                            scale(subcomm, cfg, local_B.lengths(),
                                  beta, conj_B, data_B, local_B.strides());
                        }
                    });
                }
            }

            idx_B++;
        }
        else
        {
            stride_type next_A = idx_A;

            do { next_A++; }
            while (next_A < nidx_A && get<0>(indices_A[next_A]) == get<0>(indices_B[idx_B]));

            for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
            {
                tasks.visit(task++,
                [&,idx_A,next_A,idx_B,block_AB,irreps_A,irreps_B,beta,conj_B](const communicator& subcomm)
                {
                    for (stride_type block_A = 0;block_A < nblock_A;block_A++)
                    {
                        detail::assign_irreps(dense_ndim_AB, irrep_AB, nirrep, block_AB,
                                              irreps_A, dense_idx_A_AB, irreps_B, dense_idx_B_AB);

                        detail::assign_irreps(dense_ndim_A_only, irrep_A, nirrep, block_A,
                                              irreps_A, dense_idx_A_A);

                        if (detail::is_block_empty(dpd_A, irreps_A)) continue;
                        if (detail::is_block_empty(dpd_B, irreps_B)) continue;

                        auto local_A = dpd_A(irreps_A);
                        auto local_B = dpd_B(irreps_B);

                        auto len_A = stl_ext::select_from(local_A.lengths(), dense_idx_A_A);
                        auto len_AB = stl_ext::select_from(local_A.lengths(), dense_idx_A_AB);
                        auto stride_A_A = stl_ext::select_from(local_A.strides(), dense_idx_A_A);
                        auto stride_A_AB = stl_ext::select_from(local_A.strides(), dense_idx_A_AB);
                        auto stride_B_AB = stl_ext::select_from(local_B.strides(), dense_idx_B_AB);

                        auto data_B = local_B.data() + B.data(B.data(get<1>(indices_B[idx_B]))) - B.data(0);

                        stride_type off_A = 0;
                        for (unsigned i = 0;i < mixed_idx_A_AB.size();i++)
                        {
                            off_A += get<0>(indices_A[idx_A])[mixed_pos_A_AB[i]] *
                                     local_A.stride(mixed_idx_A_AB[i]);
                        }

                        if (!mixed_idx_B_AB.empty())
                        {
                            //
                            // Pre-scale B since we will only by adding to a
                            // portion of it
                            //

                            if (beta == T(0))
                            {
                                set(subcomm, cfg, local_B.lengths(),
                                    T(0), data_B, local_B.strides());
                            }
                            else if (beta != T(1) || (is_complex<T>::value && conj_B))
                            {
                                scale(subcomm, cfg, local_B.lengths(),
                                      beta, conj_B, data_B, local_B.strides());
                            }

                            for (unsigned i = 0;i < mixed_idx_B_AB.size();i++)
                            {
                                data_B += get<0>(indices_B[idx_B])[mixed_pos_B_AB[i]] *
                                          local_B.stride(mixed_idx_B_AB[i]);
                            }

                            beta = T(1);
                            conj_B = false;
                        }

                        for (;idx_A < next_A;idx_A++)
                        {
                            auto data_A = local_A.data() + A.data(A.data(get<2>(indices_A[idx_A]))) - A.data(0) + off_A;

                            add(subcomm, cfg, len_A, {}, len_AB,
                                alpha, conj_A, data_A, stride_A_A, stride_A_AB,
                                 beta, conj_B, data_B, {}, stride_B_AB);

                            beta = T(1);
                            conj_B = false;
                        }
                    }
                });
            }

            idx_A = next_A;
            idx_B++;
        }
    }

    if (beta != T(1) || (is_complex<T>::value && conj_B))
    {
        while (idx_B < nidx_B)
        {
            for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
            {
                tasks.visit(task++,
                [&,idx_B,block_AB,irreps_B](const communicator& subcomm)
                {
                    detail::assign_irreps(dense_ndim_AB, irrep_AB, nirrep, block_AB,
                                          irreps_B, dense_idx_B_AB);

                    if (detail::is_block_empty(dpd_B, irreps_B)) continue;

                    auto local_B = dpd_B(irreps_B);

                    auto data_B = local_B.data() + B.data(B.data(get<1>(indices_B[idx_B]))) - B.data(0);

                    if (beta == T(0))
                    {
                        set(subcomm, cfg, local_B.lengths(),
                            T(0), data_B, local_B.strides());
                    }
                    else
                    {
                        scale(subcomm, cfg, local_B.lengths(),
                              beta, conj_B, data_B, local_B.strides());
                    }
                });
            }

            idx_B++;
        }
    }
}

template <typename T>
void indexed_dpd_replicate_block(const communicator& comm, const config& cfg,
                                 T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A,
                                 const dim_vector& idx_A_AB,
                                 T  beta, bool conj_B, const indexed_dpd_varray_view<      T>& B,
                                 const dim_vector& idx_B_B,
                                 const dim_vector& idx_B_AB)
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

    dim_vector dense_idx_B_B;
    dim_vector dense_idx_A_AB;
    dim_vector dense_idx_B_AB;
    dim_vector mixed_idx_A_AB;
    dim_vector mixed_idx_B_AB;
    dim_vector mixed_pos_A_AB;
    dim_vector mixed_pos_B_AB;
    dim_vector batch_idx_B_B;
    dim_vector batch_idx_A_AB;
    dim_vector batch_idx_B_AB;
    dim_vector batch_pos_A_AB;
    dim_vector batch_pos_B_AB;

    len_vector mixed_len_A_AB;
    len_vector mixed_len_B_AB;

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

    unsigned dense_ndim_B_only = 0;
    unsigned batch_ndim_B_only = 0;
    for (unsigned i : idx_B_B)
    {
        if (i < dense_ndim_B)
        {
            dense_idx_B_B.push_back(i);
            dense_ndim_B_only++;
        }
        else
        {
            batch_idx_B_B.push_back(i - dense_ndim_B);
            batch_ndim_B_only++;
        }
    }

    len_vector dense_len_A(ndim_A);
    for (unsigned i = 0;i < dense_ndim_A;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            dense_len_A[i] += A.length(i, irrep);
    }

    len_vector dense_len_B(ndim_B);
    for (unsigned i = 0;i < dense_ndim_B;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            dense_len_B[i] += B.length(i, irrep);
    }

    auto iperm_A = detail::inverse_permutation(A.permutation());
    stride_vector dense_stride_A(dense_ndim_A, 1);
    for (unsigned i = 1;i < dense_ndim_A;i++)
    {
        dense_stride_A[iperm_A[i]] = dense_stride_A[iperm_A[i-1]] *
                                     dense_len_A[iperm_A[i-1]];
    }

    auto iperm_B = detail::inverse_permutation(B.permutation());
    stride_vector dense_stride_B(dense_ndim_B, 1);
    for (unsigned i = 1;i < dense_ndim_B;i++)
    {
        dense_stride_B[iperm_B[i]] = dense_stride_B[iperm_B[i-1]] *
                                     dense_len_B[iperm_B[i-1]];
    }

    stride_type dense_B = 1;
    stride_type nblock_B = 1;
    for (unsigned i : dense_idx_B_B)
    {
        dense_B *= dense_len_B[i];
        nblock_B *= nirrep;
    }
    dense_B /= nblock_B;

    stride_type dense_AB = 1;
    stride_type nblock_AB = 1;
    for (unsigned i : dense_idx_A_AB)
    {
        dense_AB *= dense_len_A[i];
        nblock_AB *= nirrep;
    }
    dense_AB /= nblock_AB;

    if (nblock_B > 1) nblock_B /= nirrep;
    if (nblock_AB > 1) nblock_AB /= nirrep;

    unsigned irrep_AB = A.irrep();
    unsigned irrep_B = A.irrep()^B.irrep();

    auto dense_stride_B_B = stl_ext::select_from(dense_stride_B, dense_idx_B_B);
    auto dense_stride_A_AB = stl_ext::select_from(dense_stride_A, dense_idx_A_AB);
    auto dense_stride_B_AB = stl_ext::select_from(dense_stride_B, dense_idx_B_AB);

    auto reorder_B = detail::sort_by_stride(dense_stride_B_B);
    auto reorder_AB = detail::sort_by_stride(dense_stride_A_AB, dense_stride_B_AB);

    stl_ext::permute(dense_idx_B_B, reorder_B);
    stl_ext::permute(dense_idx_A_AB, reorder_AB);
    stl_ext::permute(dense_idx_B_AB, reorder_AB);

    stride_type nidx_A = A.num_indices()*stl_ext::prod(mixed_len_A_AB);
    stride_type nidx_B = B.num_indices()*stl_ext::prod(mixed_len_B_AB);

    std::vector<std::tuple<len_vector,len_type>> indices_A; indices_A.reserve(nidx_A);
    std::vector<std::tuple<len_vector,len_vector,len_type>> indices_B; indices_B.reserve(nidx_B);

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
        len_vector idx_AB(batch_ndim_AB);
        len_vector idx_B(batch_ndim_B_only);

        for (unsigned k = 0;k < batch_ndim_AB;k++)
            idx_AB[batch_pos_B_AB[k]] = B.index(i, batch_idx_B_AB[k]);

        for (unsigned k = 0;k < batch_ndim_B_only;k++)
            idx_B[k] = B.index(i, batch_idx_B_B[k]);

        while (iter_B.next())
        {
            for (unsigned k = 0;k < iter_B.dimension();k++)
                idx_AB[mixed_pos_B_AB[k]] = iter_B.position(k);

            indices_B.emplace_back(idx_AB, idx_B, i);
        }
    }

    stl_ext::sort(indices_A);
    stl_ext::sort(indices_B);

    auto dpd_A = A(0);
    auto dpd_B = B(0);

    dynamic_task_set tasks(comm, nidx_B*nblock_AB*nblock_B, dense_AB*dense_B);

    stride_type task = 0;
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
                for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
                {
                    for (stride_type block_B = 0;block_B < nblock_B;block_B++)
                    {
                        tasks.visit(task++,
                        [&,idx_B,block_AB,block_B,irreps_B](const communicator& subcomm)
                        {
                            detail::assign_irreps(dense_ndim_AB, irrep_AB, nirrep, block_AB,
                                                  irreps_B, dense_idx_B_AB);

                            detail::assign_irreps(dense_ndim_B_only, irrep_B, nirrep, block_B,
                                                  irreps_B, dense_idx_B_B);

                            if (detail::is_block_empty(dpd_B, irreps_B)) continue;

                            auto local_B = dpd_B(irreps_B);

                            auto data_B = local_B.data() + B.data(B.data(get<1>(indices_B[idx_B]))) - B.data(0);

                            if (beta == T(0))
                            {
                                set(subcomm, cfg, local_B.lengths(),
                                    T(0), data_B, local_B.strides());
                            }
                            else
                            {
                                scale(subcomm, cfg, local_B.lengths(),
                                      beta, conj_B, data_B, local_B.strides());
                            }
                        });
                    }
                }
            }

            idx_B++;
        }
        else
        {
            do
            {
                for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
                {
                    for (stride_type block_B = 0;block_B < nblock_B;block_B++)
                    {
                        tasks.visit(task++,
                        [&,idx_A,idx_B,block_AB,block_B,irreps_A,irreps_B](const communicator& subcomm)
                        {
                            detail::assign_irreps(dense_ndim_AB, irrep_AB, nirrep, block_AB,
                                                  irreps_A, dense_idx_A_AB, irreps_B, dense_idx_B_AB);

                            detail::assign_irreps(dense_ndim_B_only, irrep_B, nirrep, block_B,
                                                  irreps_B, dense_idx_B_B);

                            if (detail::is_block_empty(dpd_A, irreps_A)) continue;
                            if (detail::is_block_empty(dpd_B, irreps_B)) continue;

                            auto local_A = dpd_A(irreps_A);
                            auto local_B = dpd_B(irreps_B);

                            auto len_B = stl_ext::select_from(local_B.lengths(), dense_idx_B_B);
                            auto len_AB = stl_ext::select_from(local_A.lengths(), dense_idx_A_AB);
                            auto stride_B_B = stl_ext::select_from(local_B.strides(), dense_idx_B_B);
                            auto stride_A_AB = stl_ext::select_from(local_A.strides(), dense_idx_A_AB);
                            auto stride_B_AB = stl_ext::select_from(local_B.strides(), dense_idx_B_AB);

                            auto data_A = local_A.data() + A.data(A.data(get<1>(indices_A[idx_A]))) - A.data(0);
                            auto data_B = local_B.data() + B.data(B.data(get<2>(indices_B[idx_B]))) - B.data(0);

                            for (unsigned i = 0;i < mixed_idx_A_AB.size();i++)
                            {
                                data_A += get<0>(indices_A[idx_A])[mixed_pos_A_AB[i]] *
                                          local_A.stride(mixed_idx_A_AB[i]);
                            }

                            if (!mixed_idx_B_AB.empty())
                            {
                                //
                                // Pre-scale B since we will only by adding to a
                                // portion of it
                                //

                                if (beta == T(0))
                                {
                                    set(subcomm, cfg, local_B.lengths(),
                                        T(0), data_B, local_B.strides());
                                }
                                else if (beta != T(1) || (is_complex<T>::value && conj_B))
                                {
                                    scale(subcomm, cfg, local_B.lengths(),
                                          beta, conj_B, data_B, local_B.strides());
                                }

                                for (unsigned i = 0;i < mixed_idx_B_AB.size();i++)
                                {
                                    data_B += get<0>(indices_B[idx_B])[mixed_pos_B_AB[i]] *
                                              local_B.stride(mixed_idx_B_AB[i]);
                                }

                                add(subcomm, cfg, {}, len_B, len_AB,
                                    alpha, conj_A, data_A, {}, stride_A_AB,
                                     T(1),  false, data_B, stride_B_B, stride_B_AB);
                            }
                            else
                            {
                                add(subcomm, cfg, {}, len_B, len_AB,
                                    alpha, conj_A, data_A, {}, stride_A_AB,
                                     beta, conj_B, data_B, stride_B_B, stride_B_AB);
                            }
                        });
                    }
                }

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
            for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
            {
                for (stride_type block_B = 0;block_B < nblock_B;block_B++)
                {
                    tasks.visit(task++,
                    [&,idx_B,block_AB,block_B,irreps_B](const communicator& subcomm)
                    {
                        detail::assign_irreps(dense_ndim_AB, irrep_AB, nirrep, block_AB,
                                              irreps_B, dense_idx_B_AB);

                        detail::assign_irreps(dense_ndim_B_only, irrep_B, nirrep, block_B,
                                              irreps_B, dense_idx_B_B);

                        if (detail::is_block_empty(dpd_B, irreps_B)) continue;

                        auto local_B = dpd_B(irreps_B);

                        auto data_B = local_B.data() + B.data(B.data(get<1>(indices_B[idx_B]))) - B.data(0);

                        if (beta == T(0))
                        {
                            set(subcomm, cfg, local_B.lengths(),
                                T(0), data_B, local_B.strides());
                        }
                        else
                        {
                            scale(subcomm, cfg, local_B.lengths(),
                                  beta, conj_B, data_B, local_B.strides());
                        }
                    });
                }
            }

            idx_B++;
        }
    }
}

template <typename T>
void indexed_dpd_transpose_block(const communicator& comm, const config& cfg,
                                 T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A,
                                 const dim_vector& idx_A_AB,
                                 T  beta, bool conj_B, const indexed_dpd_varray_view<      T>& B,
                                 const dim_vector& idx_B_AB)
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

    len_vector dense_len_A(ndim_A);
    for (unsigned i = 0;i < dense_ndim_A;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            dense_len_A[i] += A.length(i, irrep);
    }

    len_vector dense_len_B(ndim_B);
    for (unsigned i = 0;i < dense_ndim_B;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            dense_len_B[i] += B.length(i, irrep);
    }

    auto iperm_A = detail::inverse_permutation(A.permutation());
    stride_vector dense_stride_A(dense_ndim_A, 1);
    for (unsigned i = 1;i < dense_ndim_A;i++)
    {
        dense_stride_A[iperm_A[i]] = dense_stride_A[iperm_A[i-1]] *
                                     dense_len_A[iperm_A[i-1]];
    }

    auto iperm_B = detail::inverse_permutation(B.permutation());
    stride_vector dense_stride_B(dense_ndim_B, 1);
    for (unsigned i = 1;i < dense_ndim_B;i++)
    {
        dense_stride_B[iperm_B[i]] = dense_stride_B[iperm_B[i-1]] *
                                     dense_len_B[iperm_B[i-1]];
    }

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
                for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
                {
                    tasks.visit(task++,
                    [&,idx_B,block_AB,irreps_B](const communicator& subcomm)
                    {
                        detail::assign_irreps(dense_ndim_AB, irrep_AB, nirrep, block_AB,
                                              irreps_B, dense_idx_B_AB);

                        if (detail::is_block_empty(dpd_B, irreps_B)) continue;

                        auto local_B = dpd_B(irreps_B);

                        auto data_B = local_B.data() + B.data(B.data(get<1>(indices_B[idx_B]))) - B.data(0);

                        if (beta == T(0))
                        {
                            set(subcomm, cfg, local_B.lengths(),
                                T(0), data_B, local_B.strides());
                        }
                        else
                        {
                            scale(subcomm, cfg, local_B.lengths(),
                                  beta, conj_B, data_B, local_B.strides());
                        }
                    });
                }
            }

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

                    if (!mixed_idx_B_AB.empty())
                    {
                        //
                        // Pre-scale B since we will only by adding to a
                        // portion of it
                        //

                        if (beta == T(0))
                        {
                            set(subcomm, cfg, local_B.lengths(),
                                T(0), data_B, local_B.strides());
                        }
                        else if (beta != T(1) || (is_complex<T>::value && conj_B))
                        {
                            scale(subcomm, cfg, local_B.lengths(),
                                  beta, conj_B, data_B, local_B.strides());
                        }

                        for (unsigned i = 0;i < mixed_idx_B_AB.size();i++)
                        {
                            data_B += get<0>(indices_B[idx_B])[mixed_pos_B_AB[i]] *
                                      local_B.stride(mixed_idx_B_AB[i]);
                        }

                        add(subcomm, cfg, {}, {}, len_AB,
                            alpha, conj_A, data_A, {}, stride_A_AB,
                             T(1),  false, data_B, {}, stride_B_AB);
                    }
                    else
                    {
                        add(subcomm, cfg, {}, {}, len_AB,
                            alpha, conj_A, data_A, {}, stride_A_AB,
                             beta, conj_B, data_B, {}, stride_B_AB);
                    }
                });
            }

            idx_A++;
            idx_B++;
        }
    }

    if (beta != T(1) || (is_complex<T>::value && conj_B))
    {
        while (idx_B < nidx_B)
        {
            for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
            {
                tasks.visit(task++,
                [&,idx_B,block_AB,irreps_B](const communicator& subcomm)
                {
                    detail::assign_irreps(dense_ndim_AB, irrep_AB, nirrep, block_AB,
                                          irreps_B, dense_idx_B_AB);

                    if (detail::is_block_empty(dpd_B, irreps_B)) continue;

                    auto local_B = dpd_B(irreps_B);

                    auto data_B = local_B.data() + B.data(B.data(get<1>(indices_B[idx_B]))) - B.data(0);

                    if (beta == T(0))
                    {
                        set(subcomm, cfg, local_B.lengths(),
                            T(0), data_B, local_B.strides());
                    }
                    else
                    {
                        scale(subcomm, cfg, local_B.lengths(),
                              beta, conj_B, data_B, local_B.strides());
                    }
                });
            }

            idx_B++;
        }
    }
}

template <typename T>
void indexed_dpd_add(const communicator& comm, const config& cfg,
                     T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A,
                     const dim_vector& idx_A_A,
                     const dim_vector& idx_A_AB,
                     T  beta, bool conj_B, const indexed_dpd_varray_view<      T>& B,
                     const dim_vector& idx_B_B,
                     const dim_vector& idx_B_AB)
{
    if (dpd_impl == FULL)
    {
        indexed_dpd_add_full(comm, cfg,
                         alpha, conj_A, A, idx_A_A, idx_A_AB,
                          beta, conj_B, B, idx_B_B, idx_B_AB);
    }
    else if (!idx_A_A.empty())
    {
        indexed_dpd_trace_block(comm, cfg,
                            alpha, conj_A, A, idx_A_A, idx_A_AB,
                             beta, conj_B, B, idx_B_AB);
    }
    else if (!idx_B_B.empty())
    {
        indexed_dpd_replicate_block(comm, cfg,
                                alpha, conj_A, A, idx_A_AB,
                                 beta, conj_B, B, idx_B_B, idx_B_AB);
    }
    else
    {
        indexed_dpd_transpose_block(comm, cfg,
                                alpha, conj_A, A, idx_A_AB,
                                 beta, conj_B, B, idx_B_AB);
    }
}

#define FOREACH_TYPE(T) \
template void indexed_dpd_add(const communicator& comm, const config& cfg, \
                              T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A, \
                              const dim_vector& idx_A, \
                              const dim_vector& idx_A_AB, \
                              T  beta, bool conj_B, const indexed_dpd_varray_view<      T>& B, \
                              const dim_vector& idx_B, \
                              const dim_vector& idx_B_AB);
#include "configs/foreach_type.h"

}
}
