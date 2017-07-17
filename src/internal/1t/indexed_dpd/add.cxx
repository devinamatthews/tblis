#include "util.hpp"
#include "add.hpp"
#include "internal/1t/dense/add.hpp"
#include "internal/1t/dense/scale.hpp"
#include "internal/1t/dense/set.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void add_full(const communicator& comm, const config& cfg,
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
            block_to_full(A, A2);
            block_to_full(B, B2);
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
            full_to_block(B2, B);
        }
    },
    A2, B2);
}

template <typename T>
void trace_block(const communicator& comm, const config& cfg,
                 T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A,
                 const dim_vector& idx_A_A,
                 const dim_vector& idx_A_AB,
                 T  beta, bool conj_B, const indexed_dpd_varray_view<      T>& B,
                 const dim_vector& idx_B_AB)
{
    unsigned nirrep = A.num_irreps();

    dpd_index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    dpd_index_group<1> group_A(A, idx_A_A);

    irrep_vector irreps_A(A.dense_dimension());
    irrep_vector irreps_B(B.dense_dimension());
    assign_irreps(group_AB, irreps_A, irreps_B);
    assign_irreps(group_A, irreps_A);

    unsigned irrep_AB = B.irrep();
    for (auto irrep : group_AB.batch_irrep) irrep_AB ^= irrep;

    unsigned irrep_A = A.irrep()^B.irrep();
    for (auto irrep : group_A.batch_irrep) irrep_A ^= irrep;

    group_indices<2> indices_A(A, group_AB, 0, group_A, 0);
    group_indices<1> indices_B(B, group_AB, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();

    auto dpd_A = A[0];
    auto dpd_B = B[0];

    dynamic_task_set tasks(comm, nidx_B*group_AB.dense_nblock, group_AB.dense_size);

    stride_type task = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    while (idx_A < nidx_A && idx_B < nidx_B)
    {
        if (indices_A[idx_A].key[0] < indices_B[idx_B].key[0])
        {
            idx_A++;
        }
        else if (indices_A[idx_A].key[0] > indices_B[idx_B].key[0])
        {
            if (beta != T(1) || (is_complex<T>::value && conj_B))
            {
                for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
                {
                    tasks.visit(task++,
                    [&,idx_B,block_AB](const communicator& subcomm)
                    {
                        auto local_irreps_B = irreps_B;

                        assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                      local_irreps_B, group_AB.dense_idx[1]);

                        if (is_block_empty(dpd_B, local_irreps_B)) return;

                        auto local_B = dpd_B(local_irreps_B);

                        auto data_B = local_B.data() + indices_B[idx_B].offset;

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
            unsigned next_A = idx_A;

            do
            {
                next_A++;
            }
            while (next_A < nidx_A && indices_A[idx_A].key[0] == indices_B[idx_B].key[0]);

            for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
            {
                tasks.visit(task++,
                [&,idx_A,idx_B,block_AB,next_A](const communicator& subcomm)
                {
                    auto local_idx_A = idx_A;
                    auto local_irreps_A = irreps_A;
                    auto local_irreps_B = irreps_B;
                    auto local_beta = beta;
                    auto local_conj_B = conj_B;

                    for (stride_type block_A = 0;block_A < group_A.dense_nblock;block_A++)
                    {
                        assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                      local_irreps_A, group_AB.dense_idx[0],
                                      local_irreps_B, group_AB.dense_idx[1]);

                        assign_irreps(group_A.dense_ndim, irrep_A, nirrep, block_A,
                                      local_irreps_A, group_A.dense_idx[0]);

                        if (is_block_empty(dpd_A, local_irreps_A)) continue;
                        if (is_block_empty(dpd_B, local_irreps_B)) continue;

                        auto local_A = dpd_A(local_irreps_A);
                        auto local_B = dpd_B(local_irreps_B);

                        len_vector len_AB;
                        stride_vector stride_A_AB, stride_B_AB;
                        stride_type off_A_AB, off_B_AB;
                        get_local_geometry(indices_A[idx_A].idx[0], group_AB, len_AB,
                                           local_A, off_A_AB, stride_A_AB, 0,
                                           local_B, off_B_AB, stride_B_AB, 1);

                        len_vector len_A;
                        stride_vector stride_A_A;
                        stride_type off_A_A;
                        get_local_geometry(indices_A[idx_A].idx[1], group_A, len_A,
                                           local_A, off_A_A, stride_A_A, 0);

                        auto data_B = local_B.data() + indices_B[idx_B].offset;

                        if (!group_AB.mixed_idx[1].empty())
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
                                      local_beta, local_conj_B, data_B, local_B.strides());
                            }

                            data_B += off_B_AB;

                            local_beta = T(1);
                            local_conj_B = false;
                        }

                        for (auto local_idx_A = idx_A;local_idx_A < next_A;local_idx_A++)
                        {
                            auto data_A = local_A.data() + indices_A[local_idx_A].offset + off_A_AB;

                            add(subcomm, cfg, len_A, {}, len_AB,
                                alpha, conj_A, data_A, stride_A_A, stride_A_AB,
                                local_beta, local_conj_B, data_B, {}, stride_B_AB);

                            local_beta = T(1);
                            local_conj_B = false;
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
            for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
            {
                tasks.visit(task++,
                [&,idx_B,block_AB](const communicator& subcomm)
                {
                    auto local_irreps_B = irreps_B;

                    assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                  local_irreps_B, group_AB.dense_idx[1]);

                    if (is_block_empty(dpd_B, local_irreps_B)) return;

                    auto local_B = dpd_B(local_irreps_B);

                    auto data_B = local_B.data() + indices_B[idx_B].offset;

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
void replicate_block(const communicator& comm, const config& cfg,
                     T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A,
                     const dim_vector& idx_A_AB,
                     T  beta, bool conj_B, const indexed_dpd_varray_view<      T>& B,
                     const dim_vector& idx_B_B,
                     const dim_vector& idx_B_AB)
{
    unsigned nirrep = A.num_irreps();

    dpd_index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    dpd_index_group<1> group_B(B, idx_B_B);

    irrep_vector irreps_A(A.dense_dimension());
    irrep_vector irreps_B(B.dense_dimension());
    assign_irreps(group_AB, irreps_A, irreps_B);
    assign_irreps(group_B, irreps_B);

    unsigned irrep_AB = A.irrep();
    for (auto irrep : group_AB.batch_irrep) irrep_AB ^= irrep;

    unsigned irrep_B = B.irrep()^irrep_AB;
    for (auto irrep : group_B.batch_irrep) irrep_B ^= irrep;

    group_indices<1> indices_A(A, group_AB, 0);
    group_indices<2> indices_B(B, group_AB, 1, group_B, 0);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();

    auto dpd_A = A[0];
    auto dpd_B = B[0];

    dynamic_task_set tasks(comm, nidx_B*group_AB.dense_nblock*group_B.dense_nblock,
                           group_AB.dense_size*group_B.dense_size);

    stride_type task = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    while (idx_A < nidx_A && idx_B < nidx_B)
    {
        if (indices_A[idx_A].key[0] < indices_B[idx_B].key[0])
        {
            idx_A++;
        }
        else if (indices_A[idx_A].key[0] > indices_B[idx_B].key[0])
        {
            if (beta != T(1) || (is_complex<T>::value && conj_B))
            {
                for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
                {
                    for (stride_type block_B = 0;block_B < group_B.dense_nblock;block_B++)
                    {
                        tasks.visit(task++,
                        [&,idx_B,block_AB,block_B](const communicator& subcomm)
                        {
                            auto local_irreps_B = irreps_B;

                            assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                          local_irreps_B, group_AB.dense_idx[1]);

                            assign_irreps(group_B.dense_ndim, irrep_B, nirrep, block_B,
                                          local_irreps_B, group_B.dense_idx[0]);

                            if (is_block_empty(dpd_B, local_irreps_B)) return;

                            auto local_B = dpd_B(local_irreps_B);

                            auto data_B = local_B.data() + indices_B[idx_B].offset;

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
                for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
                {
                    for (stride_type block_B = 0;block_B < group_B.dense_nblock;block_B++)
                    {
                        tasks.visit(task++,
                        [&,idx_A,idx_B,block_AB,block_B](const communicator& subcomm)
                        {
                            auto local_irreps_A = irreps_A;
                            auto local_irreps_B = irreps_B;

                            assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                          local_irreps_A, group_AB.dense_idx[0],
                                          local_irreps_B, group_AB.dense_idx[1]);

                            assign_irreps(group_B.dense_ndim, irrep_B, nirrep, block_B,
                                          local_irreps_B, group_B.dense_idx[0]);

                            if (is_block_empty(dpd_A, local_irreps_A)) return;
                            if (is_block_empty(dpd_B, local_irreps_B)) return;

                            auto local_A = dpd_A(local_irreps_A);
                            auto local_B = dpd_B(local_irreps_B);

                            len_vector len_AB;
                            stride_vector stride_A_AB, stride_B_AB;
                            stride_type off_A_AB, off_B_AB;
                            get_local_geometry(indices_A[idx_A].idx[0], group_AB, len_AB,
                                               local_A, off_A_AB, stride_A_AB, 0,
                                               local_B, off_B_AB, stride_B_AB, 1);

                            len_vector len_B;
                            stride_vector stride_B_B;
                            stride_type off_B_B;
                            get_local_geometry(indices_B[idx_B].idx[1], group_B, len_B,
                                               local_B, off_B_B, stride_B_B, 0);

                            auto data_A = local_A.data() + indices_A[idx_A].offset + off_A_AB;
                            auto data_B = local_B.data() + indices_B[idx_B].offset;

                            if (!group_AB.mixed_idx[1].empty())
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

                                data_B += off_B_AB;

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
            while (idx_B < nidx_B && indices_A[idx_A].key[0] == indices_B[idx_B].key[0]);

            idx_A++;
        }
    }

    if (beta != T(1) || (is_complex<T>::value && conj_B))
    {
        while (idx_B < nidx_B)
        {
            for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
            {
                for (stride_type block_B = 0;block_B < group_B.dense_nblock;block_B++)
                {
                    tasks.visit(task++,
                    [&,idx_B,block_AB,block_B](const communicator& subcomm)
                    {
                        auto local_irreps_B = irreps_B;

                        assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                      local_irreps_B, group_AB.dense_idx[1]);

                        assign_irreps(group_B.dense_ndim, irrep_B, nirrep, block_B,
                                      local_irreps_B, group_B.dense_idx[0]);

                        if (is_block_empty(dpd_B, local_irreps_B)) return;

                        auto local_B = dpd_B(local_irreps_B);

                        auto data_B = local_B.data() + indices_B[idx_B].offset;

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
void transpose_block(const communicator& comm, const config& cfg,
                     T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A,
                     const dim_vector& idx_A_AB,
                     T  beta, bool conj_B, const indexed_dpd_varray_view<      T>& B,
                     const dim_vector& idx_B_AB)
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

    stride_type task = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    while (idx_A < nidx_A && idx_B < nidx_B)
    {
        if (indices_A[idx_A].key[0] < indices_B[idx_B].key[0])
        {
            idx_A++;
        }
        else if (indices_A[idx_A].key[0] > indices_B[idx_B].key[0])
        {
            if (beta != T(1) || (is_complex<T>::value && conj_B))
            {
                for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
                {
                    tasks.visit(task++,
                    [&,idx_B,block_AB](const communicator& subcomm)
                    {
                        auto local_irreps_B = irreps_B;

                        assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                      local_irreps_B, group_AB.dense_idx[1]);

                        if (is_block_empty(dpd_B, local_irreps_B)) return;

                        auto local_B = dpd_B(local_irreps_B);

                        auto data_B = local_B.data() + indices_B[idx_B].offset;

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
            for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
            {
                tasks.visit(task++,
                [&,idx_A,idx_B,block_AB](const communicator& subcomm)
                {
                    auto local_irreps_A = irreps_A;
                    auto local_irreps_B = irreps_B;

                    assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                  local_irreps_A, group_AB.dense_idx[0],
                                  local_irreps_B, group_AB.dense_idx[1]);

                    if (is_block_empty(dpd_A, local_irreps_A)) return;
                    if (is_block_empty(dpd_B, local_irreps_B)) return;

                    auto local_A = dpd_A(local_irreps_A);
                    auto local_B = dpd_B(local_irreps_B);

                    len_vector len_AB;
                    stride_vector stride_A_AB, stride_B_AB;
                    stride_type off_A_AB, off_B_AB;
                    get_local_geometry(indices_A[idx_A].idx[0], group_AB, len_AB,
                                       local_A, off_A_AB, stride_A_AB, 0,
                                       local_B, off_B_AB, stride_B_AB, 1);

                    auto data_A = local_A.data() + indices_A[idx_A].offset + off_A_AB;
                    auto data_B = local_B.data() + indices_B[idx_B].offset;

                    if (!group_AB.mixed_idx[1].empty())
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

                        data_B += off_B_AB;

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
            for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
            {
                tasks.visit(task++,
                [&,idx_B,block_AB](const communicator& subcomm)
                {
                    auto local_irreps_B = irreps_B;

                    assign_irreps(group_AB.dense_ndim, group_AB.dense_size, nirrep, block_AB,
                                  local_irreps_B, group_AB.dense_idx[1]);

                    if (is_block_empty(dpd_B, local_irreps_B)) return;

                    auto local_B = dpd_B(local_irreps_B);

                    auto data_B = local_B.data() + indices_B[idx_B].offset;

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
void add(const communicator& comm, const config& cfg,
         T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A,
         const dim_vector& idx_A_A,
         const dim_vector& idx_A_AB,
         T  beta, bool conj_B, const indexed_dpd_varray_view<      T>& B,
         const dim_vector& idx_B_B,
         const dim_vector& idx_B_AB)
{
    if (dpd_impl == FULL)
    {
        add_full(comm, cfg,
                 alpha, conj_A, A, idx_A_A, idx_A_AB,
                  beta, conj_B, B, idx_B_B, idx_B_AB);
    }
    else if (!idx_A_A.empty())
    {
        trace_block(comm, cfg,
                    alpha, conj_A, A, idx_A_A, idx_A_AB,
                     beta, conj_B, B, idx_B_AB);
    }
    else if (!idx_B_B.empty())
    {
        replicate_block(comm, cfg,
                        alpha, conj_A, A, idx_A_AB,
                         beta, conj_B, B, idx_B_B, idx_B_AB);
    }
    else
    {
        transpose_block(comm, cfg,
                        alpha, conj_A, A, idx_A_AB,
                         beta, conj_B, B, idx_B_AB);
    }
}

#define FOREACH_TYPE(T) \
template void add(const communicator& comm, const config& cfg, \
                  T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A, \
                  const dim_vector& idx_A, \
                  const dim_vector& idx_A_AB, \
                  T  beta, bool conj_B, const indexed_dpd_varray_view<      T>& B, \
                  const dim_vector& idx_B, \
                  const dim_vector& idx_B_AB);
#include "configs/foreach_type.h"

}
}
