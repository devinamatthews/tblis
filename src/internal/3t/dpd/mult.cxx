#include "mult.hpp"
#include "internal/1t/dpd/util.hpp"
#include "internal/1t/dense/scale.hpp"
#include "internal/1t/dense/set.hpp"
#include "internal/3t/dense/mult.hpp"

#include "util/gemm_thread.hpp"
#include "util/tensor.hpp"

#include "matrix/tensor_matrix.hpp"

#include "nodes/gemm.hpp"

#include <atomic>

namespace tblis
{
namespace internal
{

std::atomic<long> flops;
dpd_impl_t dpd_impl = BLOCKED;

template <typename T>
void contract_blis(const communicator& comm, const config& cfg,
                   const len_vector& len_AB,
                   const len_vector& len_AC,
                   const len_vector& len_BC,
                   T alpha, const T* A,
                   const stride_vector& stride_A_AB,
                   const stride_vector& stride_A_AC,
                            const T* B,
                   const stride_vector& stride_B_AB,
                   const stride_vector& stride_B_BC,
                   T  beta,       T* C,
                   const stride_vector& stride_C_AC,
                   const stride_vector& stride_C_BC);

template <typename T>
void mult_full(const communicator& comm, const config& cfg,
               T alpha, const dpd_varray_view<const T>& A,
               const dim_vector& idx_A_AB,
               const dim_vector& idx_A_AC,
               const dim_vector& idx_A_ABC,
                        const dpd_varray_view<const T>& B,
               const dim_vector& idx_B_AB,
               const dim_vector& idx_B_BC,
               const dim_vector& idx_B_ABC,
               T  beta, const dpd_varray_view<      T>& C,
               const dim_vector& idx_C_AC,
               const dim_vector& idx_C_BC,
               const dim_vector& idx_C_ABC)
{
    varray<T> A2, B2, C2;

    comm.broadcast(
    [&](varray<T>& A2, varray<T>& B2, varray<T>& C2)
    {
        block_to_full(comm, cfg, A, A2);
        block_to_full(comm, cfg, B, B2);
        block_to_full(comm, cfg, C, C2);

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

        full_to_block(comm, cfg, C2, C);
    },
    A2, B2, C2);
}

template <typename T>
void contract_block(const communicator& comm, const config& cfg,
                    T alpha, dpd_varray_view<const T> A,
                    dim_vector idx_A_AB,
                    dim_vector idx_A_AC,
                             dpd_varray_view<const T> B,
                    dim_vector idx_B_AB,
                    dim_vector idx_B_BC,
                    T beta_, dpd_varray_view<      T> C,
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

    std::array<len_vector,3> len;
    std::array<stride_vector,3> stride;
    dense_total_lengths_and_strides(len, stride, A, idx_A_AB, B, idx_B_AB,
                                    C, idx_C_AC);

    auto perm_AC = detail::sort_by_stride(stl_ext::select_from(stride[2], idx_C_AC),
                                          stl_ext::select_from(stride[0], idx_A_AC));
    auto perm_BC = detail::sort_by_stride(stl_ext::select_from(stride[2], idx_C_BC),
                                          stl_ext::select_from(stride[1], idx_B_BC));
    auto perm_AB = detail::sort_by_stride(stl_ext::select_from(stride[0], idx_A_AB),
                                          stl_ext::select_from(stride[1], idx_B_AB));

    stl_ext::permute(idx_A_AC, perm_AC);
    stl_ext::permute(idx_A_AB, perm_AB);
    stl_ext::permute(idx_B_AB, perm_AB);
    stl_ext::permute(idx_B_BC, perm_BC);
    stl_ext::permute(idx_C_AC, perm_AC);
    stl_ext::permute(idx_C_BC, perm_BC);

    const bool row_major = cfg.gemm_row_major.value<T>();
    const bool transpose = row_major ? (ndim_AC > 0 ? stride[2][idx_C_AC[0]] : 0) == 1
                                     : (ndim_BC > 0 ? stride[2][idx_C_BC[0]] : 0) == 1;

    if (transpose)
    {
        using std::swap;
        swap(ndim_AC, ndim_BC);
        swap(ndim_A, ndim_B);
        swap(len[0], len[1]);
        swap(idx_A_AC, idx_B_BC);
        swap(idx_A_AB, idx_B_AB);
        swap(idx_C_AC, idx_C_BC);
        swap(A, B);
    }

    stride_type dense_AC = 1;
    stride_type nblock_AC = 1;
    for (unsigned i : idx_C_AC)
    {
        dense_AC *= len[2][i];
        nblock_AC *= nirrep;
    }
    dense_AC /= nblock_AC;

    stride_type dense_BC = 1;
    stride_type nblock_BC = 1;
    for (unsigned i : idx_C_BC)
    {
        dense_BC *= len[2][i];
        nblock_BC *= nirrep;
    }
    dense_BC /= nblock_BC;

    stride_type dense_AB = 1;
    stride_type nblock_AB = 1;
    for (unsigned i : idx_A_AB)
    {
        dense_AB *= len[0][i];
        nblock_AB *= nirrep;
    }
    dense_AB /= nblock_AB;

    stride_type nblock_par = nblock_AC*nblock_BC/nirrep;

    if (nblock_AC > 1) nblock_AC /= nirrep;
    if (nblock_BC > 1) nblock_BC /= nirrep;
    if (nblock_AB > 1) nblock_AB /= nirrep;

    stride_type block_idx = 0;

    comm.do_tasks_deferred(nblock_par, dense_AB*dense_AC*dense_BC*inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        for (unsigned irrep_AB = 0;irrep_AB < nirrep;irrep_AB++)
        {
            unsigned irrep_AC = A.irrep()^irrep_AB;
            unsigned irrep_BC = B.irrep()^irrep_AB;

            if (ndim_AC == 0 && irrep_AC != 0) continue;
            if (ndim_BC == 0 && irrep_BC != 0) continue;

            for (stride_type block_AC = 0;block_AC < nblock_AC;block_AC++)
            {
                for (stride_type block_BC = 0;block_BC < nblock_BC;block_BC++)
                {
                    tasks.visit(block_idx++,
                    [&,irrep_AB,irrep_AC,irrep_BC,block_AC,block_BC]
                    (const communicator& subcomm)
                    {
                        irrep_vector irreps_A(ndim_A);
                        irrep_vector irreps_B(ndim_B);
                        irrep_vector irreps_C(ndim_C);

                        assign_irreps(ndim_AC, irrep_AC, nirrep, block_AC,
                                      irreps_A, idx_A_AC, irreps_C, idx_C_AC);

                        assign_irreps(ndim_BC, irrep_BC, nirrep, block_BC,
                                      irreps_B, idx_B_BC, irreps_C, idx_C_BC);

                        if (is_block_empty(C, irreps_C)) return;

                        auto local_C = C(irreps_C);
                        T beta = beta_;

                        auto len_AC = stl_ext::select_from(local_C.lengths(), idx_C_AC);
                        auto len_BC = stl_ext::select_from(local_C.lengths(), idx_C_BC);
                        auto stride_C_AC = stl_ext::select_from(local_C.strides(), idx_C_AC);
                        auto stride_C_BC = stl_ext::select_from(local_C.strides(), idx_C_BC);

                        tensor_matrix<T> ct(len_AC, len_BC, local_C.data(),
                                            stride_C_AC, stride_C_BC);

                        if (ndim_AB != 0 || irrep_AB == 0)
                        {
                            auto tc = make_gemm_thread_config<T>(
                                cfg, subcomm.num_threads(), ct.length(0), ct.length(1), 0);

                            communicator comm_nc = subcomm.gang(TCI_EVENLY, tc.jc_nt);
                            communicator comm_kc = comm_nc.gang(TCI_EVENLY,        1);
                            communicator comm_mc = comm_kc.gang(TCI_EVENLY, tc.ic_nt);
                            communicator comm_nr = comm_mc.gang(TCI_EVENLY, tc.jr_nt);
                            communicator comm_mr = comm_nr.gang(TCI_EVENLY, tc.ir_nt);

                            TensorGEMM gemm;
                            step<0>(gemm).subcomm = &comm_nc;
                            step<1>(gemm).subcomm = &comm_kc;
                            step<4>(gemm).subcomm = &comm_mc;
                            step<8>(gemm).subcomm = &comm_nr;
                            step<9>(gemm).subcomm = &comm_mr;

                            for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
                            {
                                assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
                                              irreps_A, idx_A_AB, irreps_B, idx_B_AB);

                                if (is_block_empty(A, irreps_A)) continue;

                                auto local_A = A(irreps_A);
                                auto local_B = B(irreps_B);

                                auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
                                auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
                                auto stride_A_AC = stl_ext::select_from(local_A.strides(), idx_A_AC);
                                auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);
                                auto stride_B_BC = stl_ext::select_from(local_B.strides(), idx_B_BC);

                                tensor_matrix<T> at(len_AC, len_AB, const_cast<T*>(local_A.data()),
                                                    stride_A_AC, stride_A_AB);

                                tensor_matrix<T> bt(len_AB, len_BC, const_cast<T*>(local_B.data()),
                                                    stride_B_AB, stride_B_BC);

                                if (subcomm.master())
                                    flops += 2*ct.length(0)*ct.length(1)*at.length(1);

                                gemm(subcomm, cfg, alpha, at, bt, beta, ct);

                                beta = T(1);
                            }
                        }

                        if (beta == T(0))
                        {
                            set(subcomm, cfg, local_C.lengths(),
                                beta, local_C.data(), local_C.strides());
                        }
                        else if (beta != T(1))
                        {
                            scale(subcomm, cfg, local_C.lengths(),
                                  beta, false, local_C.data(), local_C.strides());
                        }
                    });
                }
            }
        }
    });
}

template <typename T>
void mult_block(const communicator& comm, const config& cfg,
                T alpha, dpd_varray_view<const T> A,
                dim_vector idx_A_AB,
                dim_vector idx_A_AC,
                dim_vector idx_A_ABC,
                         dpd_varray_view<const T> B,
                dim_vector idx_B_AB,
                dim_vector idx_B_BC,
                dim_vector idx_B_ABC,
                T beta_, dpd_varray_view<      T> C,
                dim_vector idx_C_AC,
                dim_vector idx_C_BC,
                dim_vector idx_C_ABC)
{
    unsigned nirrep = A.num_irreps();

    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();
    unsigned ndim_C = C.dimension();

    unsigned ndim_AC = idx_C_AC.size();
    unsigned ndim_BC = idx_C_BC.size();
    unsigned ndim_AB = idx_A_AB.size();
    unsigned ndim_ABC = idx_A_ABC.size();

    std::array<len_vector,3> len;
    std::array<stride_vector,3> stride;
    dense_total_lengths_and_strides(len, stride, A, idx_A_AB, B, idx_B_AB,
                                    C, idx_C_AC);

    auto perm_AC = detail::sort_by_stride(stl_ext::select_from(stride[2], idx_C_AC),
                                          stl_ext::select_from(stride[0], idx_A_AC));
    auto perm_BC = detail::sort_by_stride(stl_ext::select_from(stride[2], idx_C_BC),
                                          stl_ext::select_from(stride[1], idx_B_BC));
    auto perm_AB = detail::sort_by_stride(stl_ext::select_from(stride[0], idx_A_AB),
                                          stl_ext::select_from(stride[1], idx_B_AB));

    stl_ext::permute(idx_A_AC, perm_AC);
    stl_ext::permute(idx_A_AB, perm_AB);
    stl_ext::permute(idx_B_AB, perm_AB);
    stl_ext::permute(idx_B_BC, perm_BC);
    stl_ext::permute(idx_C_AC, perm_AC);
    stl_ext::permute(idx_C_BC, perm_BC);

    stride_type dense_AC = 1;
    stride_type nblock_AC = 1;
    for (unsigned i : idx_C_AC)
    {
        dense_AC *= len[2][i];
        nblock_AC *= nirrep;
    }
    dense_AC /= nblock_AC;

    stride_type dense_BC = 1;
    stride_type nblock_BC = 1;
    for (unsigned i : idx_C_BC)
    {
        dense_BC *= len[2][i];
        nblock_BC *= nirrep;
    }
    dense_BC /= nblock_BC;

    stride_type dense_AB = 1;
    stride_type nblock_AB = 1;
    for (unsigned i : idx_A_AB)
    {
        dense_AB *= len[0][i];
        nblock_AB *= nirrep;
    }
    dense_AB /= nblock_AB;

    stride_type dense_ABC = 1;
    stride_type nblock_ABC = 1;
    for (unsigned i : idx_A_ABC)
    {
        dense_ABC *= len[0][i];
        nblock_ABC *= nirrep;
    }
    dense_ABC /= nblock_ABC;

    if (nblock_AC > 1) nblock_AC /= nirrep;
    if (nblock_BC > 1) nblock_BC /= nirrep;
    if (nblock_AB > 1) nblock_AB /= nirrep;
    if (nblock_ABC > 1) nblock_ABC /= nirrep;

    irrep_vector irreps_A(ndim_A);
    irrep_vector irreps_B(ndim_B);
    irrep_vector irreps_C(ndim_C);

    for (unsigned irrep_ABC = 0;irrep_ABC < nirrep;irrep_ABC++)
    {
        for (unsigned irrep_AB = 0;irrep_AB < nirrep;irrep_AB++)
        {
            unsigned irrep_AC = A.irrep()^irrep_ABC^irrep_AB;
            unsigned irrep_BC = C.irrep()^irrep_ABC^irrep_AC;

            if (ndim_AC == 0 && irrep_AC != 0) continue;
            if (ndim_BC == 0 && irrep_BC != 0) continue;

            for (stride_type block_ABC = 0;block_ABC < nblock_ABC;block_ABC++)
            {
                assign_irreps(ndim_ABC, irrep_ABC, nirrep, block_ABC,
                              irreps_A, idx_A_ABC, irreps_B, idx_B_ABC, irreps_C, idx_C_ABC);

                for (stride_type block_AC = 0;block_AC < nblock_AC;block_AC++)
                {
                    assign_irreps(ndim_AC, irrep_AC, nirrep, block_AC,
                                  irreps_A, idx_A_AC, irreps_C, idx_C_AC);

                    for (stride_type block_BC = 0;block_BC < nblock_BC;block_BC++)
                    {
                        assign_irreps(ndim_BC, irrep_BC, nirrep, block_BC,
                                      irreps_B, idx_B_BC, irreps_C, idx_C_BC);

                        if (is_block_empty(C, irreps_C)) continue;

                        auto local_C = C(irreps_C);

                        auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
                        auto len_AC = stl_ext::select_from(local_C.lengths(), idx_C_AC);
                        auto len_BC = stl_ext::select_from(local_C.lengths(), idx_C_BC);
                        auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);
                        auto stride_C_AC = stl_ext::select_from(local_C.strides(), idx_C_AC);
                        auto stride_C_BC = stl_ext::select_from(local_C.strides(), idx_C_BC);

                        T beta = beta_;

                        if ((ndim_AB != 0 || irrep_AB == 0) &&
                            irrep_ABC == (A.irrep()^B.irrep()^C.irrep()))
                        {
                            for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
                            {
                                assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
                                              irreps_A, idx_A_AB, irreps_B, idx_B_AB);

                                if (is_block_empty(A, irreps_A)) continue;

                                auto local_A = A(irreps_A);
                                auto local_B = B(irreps_B);

                                auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
                                auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
                                auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
                                auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
                                auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);
                                auto stride_A_AC = stl_ext::select_from(local_A.strides(), idx_A_AC);
                                auto stride_B_BC = stl_ext::select_from(local_B.strides(), idx_B_BC);

                                mult(comm, cfg, len_AB, len_AC, len_BC, len_ABC,
                                     alpha, false, local_A.data(), stride_A_AB, stride_A_AC, stride_A_ABC,
                                            false, local_B.data(), stride_B_AB, stride_B_BC, stride_B_ABC,
                                      beta, false, local_C.data(), stride_C_AC, stride_C_BC, stride_C_ABC);

                                beta = T(1);
                            }
                        }

                        if (beta == T(0))
                        {
                            set(comm, cfg, local_C.lengths(),
                                beta, local_C.data(), local_C.strides());
                        }
                        else if (beta != T(1))
                        {
                            scale(comm, cfg, local_C.lengths(),
                                  beta, false, local_C.data(), local_C.strides());
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void mult(const communicator& comm, const config& cfg,
          T alpha, bool conj_A, const dpd_varray_view<const T>& A,
          const dim_vector& idx_A_AB,
          const dim_vector& idx_A_AC,
          const dim_vector& idx_A_ABC,
                   bool conj_B, const dpd_varray_view<const T>& B,
          const dim_vector& idx_B_AB,
          const dim_vector& idx_B_BC,
          const dim_vector& idx_B_ABC,
          T  beta, bool conj_C, const dpd_varray_view<      T>& C,
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
        contract_block(comm, cfg,
                       alpha, A, idx_A_AB, idx_A_AC,
                              B, idx_B_AB, idx_B_BC,
                        beta, C, idx_C_AC, idx_C_BC);
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
                   T alpha, bool conj_A, const dpd_varray_view<const T>& A, \
                   const dim_vector& idx_A_AB, \
                   const dim_vector& idx_A_AC, \
                   const dim_vector& idx_A_ABC, \
                            bool conj_B, const dpd_varray_view<const T>& B, \
                   const dim_vector& idx_B_AB, \
                   const dim_vector& idx_B_BC, \
                   const dim_vector& idx_B_ABC, \
                   T  beta, bool conj_C, const dpd_varray_view<      T>& C, \
                   const dim_vector& idx_C_AC, \
                   const dim_vector& idx_C_BC, \
                   const dim_vector& idx_C_ABC);
#include "configs/foreach_type.h"

}
}
