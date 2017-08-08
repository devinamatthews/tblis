#include "mult.hpp"
#include "internal/1t/indexed_dpd/util.hpp"
#include "internal/1t/indexed_dpd/set.hpp"
#include "internal/1t/indexed_dpd/scale.hpp"
#include "internal/1t/dense/scale.hpp"
#include "internal/1t/dense/set.hpp"
#include "internal/3t/dense/mult.hpp"

#include "util/gemm_thread.hpp"
#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void mult_full(const communicator& comm, const config& cfg,
               T alpha, const indexed_dpd_varray_view<const T>& A,
               const dim_vector& idx_A_AB,
               const dim_vector& idx_A_AC,
               const dim_vector& idx_A_ABC,
                        const indexed_dpd_varray_view<const T>& B,
               const dim_vector& idx_B_AB,
               const dim_vector& idx_B_BC,
               const dim_vector& idx_B_ABC,
               T  beta, const indexed_dpd_varray_view<      T>& C,
               const dim_vector& idx_C_AC,
               const dim_vector& idx_C_BC,
               const dim_vector& idx_C_ABC)
{
    varray<T> A2, B2, C2;

    comm.broadcast(
    [&](varray<T>& A2, varray<T>& B2, varray<T>& C2)
    {
        if (comm.master())
        {
            block_to_full(A, A2);
            block_to_full(B, B2);
            block_to_full(C, C2);
        }

        auto len_AB = stl_ext::select_from(A2.lengths(), idx_A_AB);
        auto len_AC = stl_ext::select_from(C2.lengths(), idx_C_AC);
        auto len_BC = stl_ext::select_from(C2.lengths(), idx_C_BC);
        auto len_ABC = stl_ext::select_from(C2.lengths(), idx_C_ABC);
        auto stride_A_AB = stl_ext::select_from(A2.strides(), idx_A_AB);
        auto stride_A_AC = stl_ext::select_from(A2.strides(), idx_A_AC);
        auto stride_B_AB = stl_ext::select_from(B2.strides(), idx_B_AB);
        auto stride_B_BC = stl_ext::select_from(B2.strides(), idx_B_BC);
        auto stride_C_AC = stl_ext::select_from(C2.strides(), idx_C_AC);
        auto stride_C_BC = stl_ext::select_from(C2.strides(), idx_C_BC);
        auto stride_A_ABC = stl_ext::select_from(A2.strides(), idx_A_ABC);
        auto stride_B_ABC = stl_ext::select_from(B2.strides(), idx_B_ABC);
        auto stride_C_ABC = stl_ext::select_from(C2.strides(), idx_C_ABC);

        mult(comm, cfg, len_AB, len_AC, len_BC, len_ABC,
             alpha, false, A2.data(), stride_A_AB, stride_A_AC, stride_A_ABC,
                    false, B2.data(), stride_B_AB, stride_B_BC, stride_B_ABC,
              beta, false, C2.data(), stride_C_AC, stride_C_BC, stride_C_ABC);

        if (comm.master())
        {
            full_to_block(C2, C);
        }
    },
    A2, B2, C2);
}

template <typename T>
void contract_block(const communicator& comm, const config& cfg,
                    T alpha, indexed_dpd_varray_view<const T> A,
                    dim_vector idx_A_AB,
                    dim_vector idx_A_AC,
                             indexed_dpd_varray_view<const T> B,
                    dim_vector idx_B_AB,
                    dim_vector idx_B_BC,
                    T  beta, indexed_dpd_varray_view<      T> C,
                    dim_vector idx_C_AC,
                    dim_vector idx_C_BC)
{
    unsigned nirrep = A.num_irreps();

    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();
    unsigned ndim_C = C.dimension();

    unsigned ndim_AC = (ndim_A+ndim_C-ndim_B)/2;
    unsigned ndim_BC = (ndim_B+ndim_C-ndim_A)/2;
    unsigned ndim_AB = (ndim_A+ndim_B-ndim_C)/2;

    dpd_index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    dpd_index_group<2> group_AC(A, idx_A_AC, C, idx_C_AC);
    dpd_index_group<2> group_BC(B, idx_B_BC, C, idx_C_BC);

    irrep_vector irreps_A(A.dense_dimension());
    irrep_vector irreps_B(B.dense_dimension());
    irrep_vector irreps_C(C.dense_dimension());
    assign_irreps(group_AB, irreps_A, irreps_B);
    assign_irreps(group_AC, irreps_A, irreps_C);
    assign_irreps(group_BC, irreps_B, irreps_C);

    group_indices<2> indices_A(A, group_AC, 0, group_AB, 0);
    group_indices<2> indices_B(B, group_BC, 0, group_AB, 1);
    group_indices<2> indices_C(C, group_AC, 1, group_BC, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();
    auto nidx_C = indices_C.size();

    auto dpd_A = A[0];
    auto dpd_B = B[0];
    auto dpd_C = C[0];

    dynamic_task_set tasks(comm, nidx_C*group_AC.dense_nblock*group_BC.dense_nblock,
                           group_AB.dense_size*group_AC.dense_size*group_BC.dense_size);

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_C = 0;

    if (beta == T(0))
    {
        set(comm, cfg, T(0), C, range(C.dense_dimension()));
    }
    else if (beta != T(1))
    {
        scale(comm, cfg, beta, false, C, range(C.dense_dimension()));
    }

    while (idx_A < nidx_A && idx_C < nidx_C)
    {
        if (indices_A[idx_A].key[0] < indices_C[idx_C].key[0])
        {
            idx_A++;
            continue;
        }
        else if (indices_A[idx_A].key[0] > indices_C[idx_C].key[0])
        {
            idx_C++;
            continue;
        }

        auto next_A = idx_A;

        do { next_A++; }
        while (next_A < nidx_A &&
               indices_A[idx_A].key[0] == indices_C[idx_C].key[0]);

        stride_type idx_B = 0;

        while (idx_B < nidx_B && idx_C < nidx_C &&
               indices_A[idx_A].key[0] == indices_C[idx_C].key[0])
        {
            if (indices_B[idx_B].key[0] < indices_C[idx_C].key[1])
            {
                idx_B++;
                continue;
            }
            else if (indices_B[idx_B].key[0] > indices_C[idx_C].key[1])
            {
                idx_C++;
                continue;
            }

            auto next_B = idx_B;

            do { next_B++; }
            while (next_B < nidx_B &&
                   indices_B[idx_B].key[0] == indices_C[idx_C].key[1]);

            for (unsigned irrep_AB = 0;irrep_AB < nirrep;irrep_AB++)
            {
                for (auto irrep : group_AB.batch_irrep) irrep_AB ^= irrep;

                unsigned irrep_AC = A.irrep()^irrep_AB;
                for (auto irrep : group_AC.batch_irrep) irrep_AC ^= irrep;

                unsigned irrep_BC = B.irrep()^irrep_AB;
                for (auto irrep : group_BC.batch_irrep) irrep_BC ^= irrep;

                if (group_AC.dense_ndim == 0 && irrep_AC != 0) continue;
                if (group_BC.dense_ndim == 0 && irrep_BC != 0) continue;

                for (stride_type block_AC = 0;block_AC < group_AC.dense_nblock;block_AC++)
                for (stride_type block_BC = 0;block_BC < group_BC.dense_nblock;block_BC++)
                {
                    tasks.visit(idx++,
                    [&,idx_A,irreps_A,idx_B,irreps_B,idx_C,irreps_C,
                     next_A,next_B,irrep_AB,irrep_AC,irrep_BC,block_AC,block_BC,beta]
                    (const communicator& subcomm)
                    {
                        auto local_idx_A = idx_A;
                        auto local_idx_B = idx_B;
                        auto local_irreps_A = irreps_A;
                        auto local_irreps_B = irreps_B;
                        auto local_irreps_C = irreps_C;

                        assign_irreps(group_AC.dense_ndim, irrep_AC, nirrep, block_AC,
                                      local_irreps_A, group_AC.dense_idx[0],
                                      local_irreps_C, group_AC.dense_idx[1]);

                        assign_irreps(group_BC.dense_ndim, irrep_BC, nirrep, block_BC,
                                      local_irreps_B, group_BC.dense_idx[0],
                                      local_irreps_C, group_BC.dense_idx[1]);

                        if (is_block_empty(dpd_C, local_irreps_C)) return;

                        auto local_C = dpd_C(local_irreps_C);

                        if (group_AB.dense_ndim != 0 || irrep_AB == 0)
                        {
                            for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
                            {
                                assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                              local_irreps_A, group_AB.dense_idx[0],
                                              local_irreps_B, group_AB.dense_idx[1]);

                                if (is_block_empty(dpd_A, local_irreps_A)) return;

                                auto local_A = dpd_A(local_irreps_A);
                                auto local_B = dpd_B(local_irreps_B);

                                len_vector len_AC;
                                stride_vector stride_A_AC, stride_C_AC;
                                stride_type off_A_AC, off_C_AC;
                                get_local_geometry(indices_A[local_idx_A].idx[0], group_AC, len_AC,
                                                   local_A, off_A_AC, stride_A_AC, 0,
                                                   local_C, off_C_AC, stride_C_AC, 1);

                                len_vector len_BC;
                                stride_vector stride_B_BC, stride_C_BC;
                                stride_type off_B_BC, off_C_BC;
                                get_local_geometry(indices_B[local_idx_B].idx[0], group_BC, len_BC,
                                                   local_B, off_B_BC, stride_B_BC, 0,
                                                   local_C, off_C_BC, stride_C_BC, 1);

                                auto data_C = local_C.data() + indices_C[idx_C].offset + off_C_AC + off_C_BC;

                                while (local_idx_A < next_A && local_idx_B < next_B)
                                {
                                    if (indices_A[local_idx_A].key[1] < indices_B[local_idx_B].key[1])
                                    {
                                        local_idx_A++;
                                    }
                                    else if (indices_A[local_idx_A].key[1] > indices_B[local_idx_B].key[1])
                                    {
                                        local_idx_B++;
                                    }
                                    else
                                    {
                                        len_vector len_AB;
                                        stride_vector stride_A_AB, stride_B_AB;
                                        stride_type off_A_AB, off_B_AB;
                                        get_local_geometry(indices_A[local_idx_A].idx[1], group_AB, len_AB,
                                                           local_A, off_A_AB, stride_A_AB, 0,
                                                           local_B, off_B_AB, stride_B_AB, 1);

                                        auto data_A = local_A.data() + indices_A[local_idx_A].offset + off_A_AB + off_A_AC;
                                        auto data_B = local_B.data() + indices_B[local_idx_B].offset + off_B_AB + off_B_BC;

                                        mult(subcomm, cfg,
                                             len_AC, len_BC, len_AB, {},
                                             alpha, false, data_A, stride_A_AC, stride_A_AB, {},
                                                    false, data_B, stride_B_BC, stride_B_AB, {},
                                              T(1), false, data_C, stride_C_AC, stride_C_BC, {});
                                    }
                                }
                            }
                        }
                    });
                }
            }

            idx_B = next_B;
            idx_C++;
        }

        idx_A = next_A;
        idx_C++;
    }
}

template <typename T>
void mult_block(const communicator& comm, const config& cfg,
                T alpha, indexed_dpd_varray_view<const T> A,
                dim_vector idx_A_AB,
                dim_vector idx_A_AC,
                dim_vector idx_A_ABC,
                         indexed_dpd_varray_view<const T> B,
                dim_vector idx_B_AB,
                dim_vector idx_B_BC,
                dim_vector idx_B_ABC,
                T  beta, indexed_dpd_varray_view<      T> C,
                dim_vector idx_C_AC,
                dim_vector idx_C_BC,
                dim_vector idx_C_ABC)
{
    unsigned nirrep = A.num_irreps();

    unsigned ndim_ABC = idx_C_ABC.size();
    unsigned ndim_AC = idx_C_AC.size();
    unsigned ndim_BC = idx_C_BC.size();
    unsigned ndim_AB = idx_A_AB.size();

    dpd_index_group<3> group_ABC(A, idx_A_ABC, B, idx_B_ABC, C, idx_C_ABC);
    dpd_index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    dpd_index_group<2> group_AC(A, idx_A_AC, C, idx_C_AC);
    dpd_index_group<2> group_BC(B, idx_B_BC, C, idx_C_BC);

    irrep_vector irreps_A(A.dense_dimension());
    irrep_vector irreps_B(B.dense_dimension());
    irrep_vector irreps_C(C.dense_dimension());
    assign_irreps(group_ABC, irreps_A, irreps_B, irreps_C);
    assign_irreps(group_AB, irreps_A, irreps_B);
    assign_irreps(group_AC, irreps_A, irreps_C);
    assign_irreps(group_BC, irreps_B, irreps_C);

    group_indices<3> indices_A(A, group_ABC, 0, group_AC, 0, group_AB, 0);
    group_indices<3> indices_B(B, group_ABC, 1, group_BC, 0, group_AB, 1);
    group_indices<3> indices_C(C, group_ABC, 2, group_AC, 1, group_BC, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();
    auto nidx_C = indices_C.size();

    auto dpd_A = A[0];
    auto dpd_B = B[0];
    auto dpd_C = C[0];

    dynamic_task_set tasks(comm, nidx_C*group_AC.dense_nblock*group_BC.dense_nblock*group_ABC.dense_nblock,
                           group_AB.dense_size*group_AC.dense_size*group_BC.dense_size*group_ABC.dense_size);

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_B0 = 0;
    stride_type idx_C = 0;

    if (beta == T(0))
    {
        set(comm, cfg, T(0), C, range(C.dense_dimension()));
    }
    else if (beta != T(1))
    {
        scale(comm, cfg, beta, false, C, range(C.dense_dimension()));
    }

    unsigned irrep_ABC = A.irrep()^B.irrep()^C.irrep();
    for (auto irrep : group_ABC.batch_irrep) irrep_ABC ^= irrep;

    while (idx_A < nidx_A && idx_B0 < nidx_B && idx_C < nidx_C)
    {
        if (indices_A[idx_A].key[0] < indices_B[idx_B0].key[0])
        {
            if (indices_A[idx_A].key[0] < indices_C[idx_C].key[0])
            {
                idx_A++;
            }
            else
            {
                idx_C++;
            }
            continue;
        }
        else if (indices_A[idx_A].key[0] > indices_B[idx_B0].key[0])
        {
            if (indices_B[idx_B0].key[0] < indices_C[idx_C].key[0])
            {
                idx_B0++;
            }
            else
            {
                idx_C++;
            }
            continue;
        }

        if (indices_A[idx_A].key[0] < indices_C[idx_C].key[0])
        {
            idx_A++;
            idx_B0++;
            continue;
        }
        else if (indices_A[idx_A].key[0] > indices_C[idx_C].key[0])
        {
            idx_C++;
            continue;
        }

        auto next_A_ABC = idx_A;
        auto next_B_ABC = idx_B0;
        auto next_C_ABC = idx_C;

        do { next_A_ABC++; }
        while (next_A_ABC < nidx_A &&
               indices_A[next_A_ABC].key[0] == indices_A[idx_A].key[0]);

        do { next_B_ABC++; }
        while (next_B_ABC < nidx_B &&
               indices_B[next_B_ABC].key[0] == indices_B[idx_B0].key[0]);

        do { next_C_ABC++; }
        while (next_C_ABC < nidx_C &&
               indices_C[next_C_ABC].key[0] == indices_C[idx_C].key[0]);

        while (idx_A < next_A_ABC && idx_C < next_C_ABC)
        {
            if (indices_A[idx_A].key[1] < indices_C[idx_C].key[1])
            {
                idx_A++;
                continue;
            }
            else if (indices_A[idx_A].key[1] > indices_C[idx_C].key[1])
            {
                idx_C++;
                continue;
            }

            auto next_A_AB = idx_A;

            do { next_A_AB++; }
            while (next_A_AB < next_A_ABC &&
                   indices_A[next_A_AB].key[1] == indices_A[idx_A].key[1]);

            stride_type idx_B = idx_B0;

            while (idx_B < next_B_ABC && idx_C < next_C_ABC &&
                   indices_A[idx_A].key[1] == indices_C[idx_C].key[1])
            {
                if (indices_B[idx_B].key[1] < indices_C[idx_C].key[2])
                {
                    idx_B++;
                    continue;
                }
                else if (indices_B[idx_B].key[1] > indices_C[idx_C].key[2])
                {
                    idx_C++;
                    continue;
                }

                auto next_B_AB = idx_B;

                do { next_B_AB++; }
                while (next_B_AB < next_B_ABC &&
                       indices_B[next_B_AB].key[1] == indices_B[idx_B].key[1]);

                for (unsigned irrep_AB = 0;irrep_AB < nirrep;irrep_AB++)
                {
                    for (auto irrep : group_AB.batch_irrep) irrep_AB ^= irrep;

                    unsigned irrep_AC = A.irrep()^irrep_AB^irrep_ABC;
                    for (auto irrep : group_AC.batch_irrep) irrep_AC ^= irrep;

                    unsigned irrep_BC = B.irrep()^irrep_AB^irrep_ABC;
                    for (auto irrep : group_BC.batch_irrep) irrep_BC ^= irrep;

                    if (group_AC.dense_ndim == 0 && irrep_AC != 0) continue;
                    if (group_BC.dense_ndim == 0 && irrep_BC != 0) continue;

                    for (stride_type block_ABC = 0;block_ABC < group_AC.dense_nblock;block_ABC++)
                    for (stride_type block_AC = 0;block_AC < group_AC.dense_nblock;block_AC++)
                    for (stride_type block_BC = 0;block_BC < group_BC.dense_nblock;block_BC++)
                    {
                        tasks.visit(idx++,
                        [&,idx_A,idx_B,idx_C,next_A_AB,next_B_AB,
                         irrep_AB,irrep_AC,irrep_BC,block_AC,block_BC,block_ABC]
                        (const communicator& subcomm)
                        {
                            auto local_idx_A = idx_A;
                            auto local_idx_B = idx_B;
                            auto local_irreps_A = irreps_A;
                            auto local_irreps_B = irreps_B;
                            auto local_irreps_C = irreps_C;

                            assign_irreps(ndim_ABC, irrep_ABC, nirrep, block_ABC,
                                          local_irreps_A, idx_A_ABC,
                                          local_irreps_B, idx_B_ABC,
                                          local_irreps_C, idx_C_ABC);

                            assign_irreps(group_AC.dense_ndim, irrep_AC, nirrep, block_AC,
                                          local_irreps_A, group_AC.dense_idx[0],
                                          local_irreps_C, group_AC.dense_idx[1]);

                            assign_irreps(group_BC.dense_ndim, irrep_BC, nirrep, block_BC,
                                          local_irreps_B, group_BC.dense_idx[0],
                                          local_irreps_C, group_BC.dense_idx[1]);

                            if (is_block_empty(dpd_C, local_irreps_C)) return;

                            auto local_C = dpd_C(local_irreps_C);

                            if (group_AB.dense_ndim != 0 || irrep_AB == 0)
                            {
                                for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
                                {
                                    assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                                  local_irreps_A, group_AB.dense_idx[0],
                                                  local_irreps_B, group_AB.dense_idx[1]);

                                    if (is_block_empty(dpd_A, local_irreps_A)) continue;

                                    auto local_A = dpd_A(local_irreps_A);
                                    auto local_B = dpd_B(local_irreps_B);

                                    len_vector len_ABC;
                                    stride_vector stride_A_ABC, stride_B_ABC, stride_C_ABC;
                                    stride_type off_A_ABC, off_B_ABC, off_C_ABC;
                                    get_local_geometry(indices_A[local_idx_A].idx[0], group_ABC, len_ABC,
                                                       local_A, off_A_ABC, stride_A_ABC, 0,
                                                       local_B, off_B_ABC, stride_B_ABC, 1,
                                                       local_C, off_C_ABC, stride_C_ABC, 2);

                                    len_vector len_AC;
                                    stride_vector stride_A_AC, stride_C_AC;
                                    stride_type off_A_AC, off_C_AC;
                                    get_local_geometry(indices_A[local_idx_A].idx[1], group_AC, len_AC,
                                                       local_A, off_A_AC, stride_A_AC, 0,
                                                       local_C, off_C_AC, stride_C_AC, 1);

                                    len_vector len_BC;
                                    stride_vector stride_B_BC, stride_C_BC;
                                    stride_type off_B_BC, off_C_BC;
                                    get_local_geometry(indices_B[local_idx_B].idx[1], group_BC, len_BC,
                                                       local_B, off_B_BC, stride_B_BC, 0,
                                                       local_C, off_C_BC, stride_C_BC, 1);

                                    auto data_C = local_C.data() + indices_C[idx_C].offset +
                                        off_C_AC + off_C_BC + off_C_ABC;

                                    while (local_idx_A < next_A_AB && local_idx_B < next_B_AB)
                                    {
                                        if (indices_A[local_idx_A].key[2] < indices_B[local_idx_B].key[2])
                                        {
                                            local_idx_A++;
                                        }
                                        else if (indices_A[local_idx_A].key[2] > indices_B[local_idx_B].key[2])
                                        {
                                            local_idx_B++;
                                        }
                                        else
                                        {
                                            len_vector len_AB;
                                            stride_vector stride_A_AB, stride_B_AB;
                                            stride_type off_A_AB, off_B_AB;
                                            get_local_geometry(indices_A[local_idx_A].idx[2], group_AB, len_AB,
                                                               local_A, off_A_AB, stride_A_AB, 0,
                                                               local_B, off_B_AB, stride_B_AB, 1);

                                            auto data_A = local_A.data() + indices_A[local_idx_A].offset +
                                                          off_A_AB + off_A_AC + off_A_ABC;
                                            auto data_B = local_B.data() + indices_B[local_idx_B].offset +
                                                          off_B_AB + off_B_BC + off_B_ABC;

                                            mult(subcomm, cfg,
                                                 len_AC, len_BC, len_AB, len_ABC,
                                                 alpha, false, data_A, stride_A_AC, stride_A_AB, stride_A_ABC,
                                                        false, data_B, stride_B_BC, stride_B_AB, stride_B_ABC,
                                                  T(1), false, data_C, stride_C_AC, stride_C_BC, stride_C_ABC);
                                        }
                                    }
                                }
                            }
                        });
                    }
                }

                idx_B = next_B_AB;
                idx_C++;
            }

            idx_A = next_A_AB;
            idx_C++;
        }

        idx_A = next_A_ABC;
        idx_B0 = next_B_ABC;
        idx_C = next_C_ABC;
    }
}

template <typename T>
void mult(const communicator& comm, const config& cfg,
          T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A,
          const dim_vector& idx_A_AB,
          const dim_vector& idx_A_AC,
          const dim_vector& idx_A_ABC,
                   bool conj_B, const indexed_dpd_varray_view<const T>& B,
          const dim_vector& idx_B_AB,
          const dim_vector& idx_B_BC,
          const dim_vector& idx_B_ABC,
          T  beta, bool conj_C, const indexed_dpd_varray_view<      T>& C,
          const dim_vector& idx_C_AC,
          const dim_vector& idx_C_BC,
          const dim_vector& idx_C_ABC)
{
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C);


    if (dpd_impl == FULL)
    {
        mult_full(comm, cfg,
                  alpha, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                         B, idx_B_AB, idx_B_BC, idx_B_ABC,
                   beta, C, idx_C_AC, idx_C_BC, idx_C_ABC);
    }
    else if (idx_C_ABC.empty())
    {
        if (A.irrep()^B.irrep()^C.irrep())
        {
            dim_vector idx_C = range(C.dimension());

            if (beta == T(0))
            {
                set(comm, cfg, T(0), C, idx_C);
            }
            else if (beta != T(1) || (is_complex<T>::value && conj_C))
            {
                scale(comm, cfg, beta, conj_C, C, idx_C);
            }
        }
        else
        {
            contract_block(comm, cfg,
                           alpha, A, idx_A_AB, idx_A_AC,
                                  B, idx_B_AB, idx_B_BC,
                            beta, C, idx_C_AC, idx_C_BC);
        }
    }
    else
    {
        mult_block(comm, cfg,
                   alpha, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                          B, idx_B_AB, idx_B_BC, idx_B_ABC,
                    beta, C, idx_C_AC, idx_C_BC, idx_C_ABC);
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, const config& cfg, \
                   T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A, \
                   const dim_vector& idx_A_AB, \
                   const dim_vector& idx_A_AC, \
                   const dim_vector& idx_A_ABC, \
                            bool conj_B, const indexed_dpd_varray_view<const T>& B, \
                   const dim_vector& idx_B_AB, \
                   const dim_vector& idx_B_BC, \
                   const dim_vector& idx_B_ABC, \
                   T  beta, bool conj_C, const indexed_dpd_varray_view<      T>& C, \
                   const dim_vector& idx_C_AC, \
                   const dim_vector& idx_C_BC, \
                   const dim_vector& idx_C_ABC);
#include "configs/foreach_type.h"

}
}
