#include "dpd_mult.hpp"
#include "mult.hpp"

#include "nodes/gemm.hpp"

#include "matrix/scatter_tensor_matrix.hpp"

#include "util/gemm_thread.hpp"
#include "util/tensor.hpp"

#include <atomic>
#include <numeric>
#include <functional>

namespace tblis
{
namespace internal
{

std::atomic<long> flops;

template <typename T>
void contract_blis(const communicator& comm, const config& cfg,
                   const std::vector<len_type>& len_AB,
                   const std::vector<len_type>& len_AC,
                   const std::vector<len_type>& len_BC,
                   T alpha, const T* A,
                   const std::vector<stride_type>& stride_A_AB,
                   const std::vector<stride_type>& stride_A_AC,
                            const T* B,
                   const std::vector<stride_type>& stride_B_AB,
                   const std::vector<stride_type>& stride_B_BC,
                   T  beta,       T* C,
                   const std::vector<stride_type>& stride_C_AC,
                   const std::vector<stride_type>& stride_C_BC);

template <typename T>
void dpd_contract_block(const communicator& comm, const config& cfg,
                        T alpha, dpd_varray_view<const T> A,
                        std::vector<unsigned> idx_A_AB,
                        std::vector<unsigned> idx_A_AC,
                                 dpd_varray_view<const T> B,
                        std::vector<unsigned> idx_B_AB,
                        std::vector<unsigned> idx_B_BC,
                        T beta_, dpd_varray_view<      T> C,
                        std::vector<unsigned> idx_C_AC,
                        std::vector<unsigned> idx_C_BC)
{
    using stl_ext::intersection;
    using stl_ext::exclusion;
    using stl_ext::select_from;
    using stl_ext::permute;

    unsigned nirrep = A.num_irreps();

    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();
    unsigned ndim_C = C.dimension();

    unsigned ndim_AC = (ndim_A+ndim_C-ndim_B)/2;
    unsigned ndim_BC = (ndim_B+ndim_C-ndim_A)/2;
    unsigned ndim_AB = (ndim_A+ndim_B-ndim_C)/2;

    std::vector<len_type> len_A(ndim_A);
    for (unsigned i = 0;i < ndim_A;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            len_A[i] += A.length(i, irrep);
    }

    std::vector<len_type> len_B(ndim_B);
    for (unsigned i = 0;i < ndim_B;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            len_B[i] += B.length(i, irrep);
    }

    std::vector<len_type> len_C(ndim_C);
    for (unsigned i = 0;i < ndim_C;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            len_C[i] += C.length(i, irrep);
    }

    auto iperm_A = detail::inverse_permutation(A.permutation());
    std::vector<stride_type> stride_A(ndim_A, 1);
    for (unsigned i = 1;i < ndim_A;i++)
    {
        stride_A[iperm_A[i]] = stride_A[iperm_A[i-1]]*len_A[iperm_A[i-1]];
    }

    auto iperm_B = detail::inverse_permutation(B.permutation());
    std::vector<stride_type> stride_B(ndim_B, 1);
    for (unsigned i = 1;i < ndim_B;i++)
    {
        stride_B[iperm_B[i]] = stride_B[iperm_B[i-1]]*len_B[iperm_B[i-1]];
    }

    auto iperm_C = detail::inverse_permutation(C.permutation());
    std::vector<stride_type> stride_C(ndim_C, 1);
    for (unsigned i = 1;i < ndim_C;i++)
    {
        stride_C[iperm_C[i]] = stride_C[iperm_C[i-1]]*len_C[iperm_C[i-1]];
    }

    auto perm_AC = detail::sort_by_stride(select_from(stride_C, idx_C_AC),
                                          select_from(stride_A, idx_A_AC));
    auto perm_BC = detail::sort_by_stride(select_from(stride_C, idx_C_BC),
                                          select_from(stride_B, idx_B_BC));
    auto perm_AB = detail::sort_by_stride(select_from(stride_A, idx_A_AB),
                                          select_from(stride_B, idx_B_AB));

    permute(idx_A_AC, perm_AC);
    permute(idx_A_AB, perm_AB);
    permute(idx_B_AB, perm_AB);
    permute(idx_B_BC, perm_BC);
    permute(idx_C_AC, perm_AC);
    permute(idx_C_BC, perm_BC);

    const bool row_major = cfg.gemm_row_major.value<T>();
    const bool transpose = row_major ? (ndim_AC > 0 ? stride_C[idx_C_AC[0]] : 0) == 1
                                     : (ndim_BC > 0 ? stride_C[idx_C_BC[0]] : 0) == 1;

    if (transpose)
    {
        using std::swap;
        swap(ndim_AC, ndim_BC);
        swap(ndim_A, ndim_B);
        swap(len_A, len_B);
        swap(idx_A_AC, idx_B_BC);
        swap(idx_A_AB, idx_B_AB);
        swap(idx_C_AC, idx_C_BC);
        swap(A, B);
    }

    stride_type dense_AC = 1;
    stride_type nblock_AC = 1;
    for (unsigned i : idx_C_AC)
    {
        dense_AC *= len_A[i];
        nblock_AC *= nirrep;
    }
    dense_AC /= nblock_AC;

    stride_type dense_BC = 1;
    stride_type nblock_BC = 1;
    for (unsigned i : idx_C_BC)
    {
        dense_BC *= len_B[i];
        nblock_BC *= nirrep;
    }
    dense_BC /= nblock_BC;

    stride_type dense_AB = 1;
    stride_type nblock_AB = 1;
    for (unsigned i : idx_A_AB)
    {
        dense_AB *= len_A[i];
        nblock_AB *= nirrep;
    }
    dense_AB /= nblock_AB;

    stride_type nblock_par = nblock_AC*nblock_BC/nirrep;
    if (ndim_AB == 0) nblock_par /= nirrep;

    if (nblock_AC > 1) nblock_AC /= nirrep;
    if (nblock_BC > 1) nblock_BC /= nirrep;
    if (nblock_AB > 1) nblock_AB /= nirrep;

    dynamic_task_set tasks(comm, nblock_par, dense_AC*dense_BC);

    stride_type block_idx = 0;
    for (unsigned irrep_AB = 0;irrep_AB < nirrep;irrep_AB++)
    {
        unsigned irrep_AC = A.irrep()^irrep_AB;
        unsigned irrep_BC = B.irrep()^irrep_AB;

        if (ndim_AC == 0 && irrep_AC != 0) continue;
        if (ndim_BC == 0 && irrep_BC != 0) continue;
        if (ndim_AB == 0 && irrep_AB != 0) continue;

        for (stride_type block_AC = 0;block_AC < nblock_AC;block_AC++)
        {
            for (stride_type block_BC = 0;block_BC < nblock_BC;block_BC++)
            {
                tasks.visit(block_idx++,
                [&,irrep_AB,irrep_AC,irrep_BC,block_AC,block_BC]
                (const communicator& subcomm, int)
                {
                    std::vector<unsigned> irreps_A(ndim_A);
                    std::vector<unsigned> irreps_B(ndim_B);
                    std::vector<unsigned> irreps_C(ndim_C);

                    detail::assign_irreps(ndim_AC, irrep_AC, nirrep, block_AC,
                                  irreps_A, idx_A_AC, irreps_C, idx_C_AC);

                    detail::assign_irreps(ndim_BC, irrep_BC, nirrep, block_BC,
                                  irreps_B, idx_B_BC, irreps_C, idx_C_BC);

                    tensor_matrix<T> ct(C(irreps_C), idx_C_AC, idx_C_BC);
                    T beta = beta_;

                    for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
                    {
                        detail::assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
                                      irreps_A, idx_A_AB, irreps_B, idx_B_AB);

                        tensor_matrix<T> at(A(irreps_A), idx_A_AC, idx_A_AB);
                        tensor_matrix<T> bt(B(irreps_B), idx_B_AB, idx_B_BC);

                        if (subcomm.master())
                            flops += 2*ct.length(0)*ct.length(1)*at.length(1);

                        TensorGEMM gemm;

                        auto tc = make_gemm_thread_config<T>(
                            cfg, subcomm.num_threads(), ct.length(0), ct.length(1), at.length(1));

                        step<0>(gemm).distribute = tc.jc_nt;
                        step<4>(gemm).distribute = tc.ic_nt;
                        step<8>(gemm).distribute = tc.jr_nt;
                        step<9>(gemm).distribute = tc.ir_nt;

                        gemm(subcomm, cfg, alpha, at, bt, beta, ct);

                        beta = T(1);
                    }
                });
            }
        }
    }
}

template <typename T>
void dpd_mult_ref(const communicator& comm, const config& cfg,
                  T alpha, const dpd_varray_view<const T>& A,
                  const std::vector<unsigned>& idx_A_only,
                  const std::vector<unsigned>& idx_A_AB,
                  const std::vector<unsigned>& idx_A_AC,
                  const std::vector<unsigned>& idx_A_ABC,
                           const dpd_varray_view<const T>& B,
                  const std::vector<unsigned>& idx_B_only,
                  const std::vector<unsigned>& idx_B_AB,
                  const std::vector<unsigned>& idx_B_BC,
                  const std::vector<unsigned>& idx_B_ABC,
                  T  beta, const dpd_varray_view<      T>& C,
                  const std::vector<unsigned>& idx_C_only,
                  const std::vector<unsigned>& idx_C_AC,
                  const std::vector<unsigned>& idx_C_BC,
                  const std::vector<unsigned>& idx_C_ABC)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();
    unsigned ndim_C = C.dimension();

    std::vector<len_type> len_A(ndim_A);
    matrix<len_type> off_A({ndim_A, nirrep});
    for (unsigned i = 0;i < ndim_A;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            off_A[i][irrep] = len_A[i];
            len_A[i] += A.length(i, irrep);
        }
    }

    std::vector<len_type> len_B(ndim_B);
    matrix<len_type> off_B({ndim_B, nirrep});
    for (unsigned i = 0;i < ndim_B;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            off_B[i][irrep] = len_B[i];
            len_B[i] += B.length(i, irrep);
        }
    }

    std::vector<len_type> len_C(ndim_C);
    matrix<len_type> off_C({ndim_C, nirrep});
    for (unsigned i = 0;i < ndim_C;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            off_C[i][irrep] = len_C[i];
            len_C[i] += C.length(i, irrep);
        }
    }

    varray<T> A2, B2, C2;

    comm.broadcast_nowait(
    [&](varray<T>& A2, varray<T>& B2, varray<T>& C2)
    {
        if (comm.master())
        {
            A2.reset(len_A);
            B2.reset(len_B);
            C2.reset(len_C);

            A.for_each_block(
            [&](const varray_view<const T>& local_A, const std::vector<unsigned>& irreps_A)
            {
                varray_view<T> local_A2 = A2;

                for (unsigned i = 0;i < ndim_A;i++)
                {
                    local_A2.length(i, local_A.length(i));
                    local_A2.shift(i, off_A[i][irreps_A[i]]);
                }

                local_A2 = local_A;
            });

            B.for_each_block(
            [&](const varray_view<const T>& local_B, const std::vector<unsigned>& irreps_B)
            {
                varray_view<T> local_B2 = B2;

                for (unsigned i = 0;i < ndim_B;i++)
                {
                    local_B2.length(i, local_B.length(i));
                    local_B2.shift(i, off_B[i][irreps_B[i]]);
                }

                local_B2 = local_B;
            });

            C.for_each_block(
            [&](const varray_view<T>& local_C, const std::vector<unsigned>& irreps_C)
            {
                varray_view<T> local_C2 = C2;

                for (unsigned i = 0;i < ndim_C;i++)
                {
                    local_C2.length(i, local_C.length(i));
                    local_C2.shift(i, off_C[i][irreps_C[i]]);
                }

                local_C2 = local_C;
            });
        }

        auto len_A_only = stl_ext::select_from(A2.lengths(), idx_A_only);
        auto len_B_only = stl_ext::select_from(B2.lengths(), idx_B_only);
        auto len_C_only = stl_ext::select_from(C2.lengths(), idx_C_only);
        auto len_AB = stl_ext::select_from(A2.lengths(), idx_A_AB);
        auto len_AC = stl_ext::select_from(C2.lengths(), idx_C_AC);
        auto len_BC = stl_ext::select_from(C2.lengths(), idx_C_BC);
        auto len_ABC = stl_ext::select_from(C2.lengths(), idx_C_ABC);
        auto stride_A_A = stl_ext::select_from(A2.strides(), idx_A_only);
        auto stride_B_B = stl_ext::select_from(B2.strides(), idx_B_only);
        auto stride_C_C = stl_ext::select_from(C2.strides(), idx_C_only);
        auto stride_A_AB = stl_ext::select_from(A2.strides(), idx_A_AB);
        auto stride_A_AC = stl_ext::select_from(A2.strides(), idx_A_AC);
        auto stride_B_AB = stl_ext::select_from(B2.strides(), idx_B_AB);
        auto stride_B_BC = stl_ext::select_from(B2.strides(), idx_B_BC);
        auto stride_C_AC = stl_ext::select_from(C2.strides(), idx_C_AC);
        auto stride_C_BC = stl_ext::select_from(C2.strides(), idx_C_BC);
        auto stride_A_ABC = stl_ext::select_from(A2.strides(), idx_A_ABC);
        auto stride_B_ABC = stl_ext::select_from(B2.strides(), idx_B_ABC);
        auto stride_C_ABC = stl_ext::select_from(C2.strides(), idx_C_ABC);

        mult(comm, cfg, len_A_only, len_B_only, len_C_only, len_AB, len_AC, len_BC, len_ABC,
             alpha, false, A2.data(), stride_A_A, stride_A_AB, stride_A_AC, stride_A_ABC,
                    false, B2.data(), stride_B_B, stride_B_AB, stride_B_BC, stride_B_ABC,
              beta, false, C2.data(), stride_C_C, stride_C_AC, stride_C_BC, stride_C_ABC);

        if (comm.master())
        {
            C.for_each_block(
            [&](const varray_view<T>& local_C, const std::vector<unsigned>& irreps_C)
            {
                varray_view<T> local_C2 = C2;

                for (unsigned i = 0;i < ndim_C;i++)
                {
                    local_C2.length(i, local_C.length(i));
                    local_C2.shift(i, off_C[i][irreps_C[i]]);
                }

                local_C = local_C2;
            });
        }
    },
    A2, B2, C2);
}

template <typename T>
void dpd_mult_block(const communicator& comm, const config& cfg,
                    T alpha, dpd_varray_view<const T> A,
                    std::vector<unsigned> idx_A_only,
                    std::vector<unsigned> idx_A_AB,
                    std::vector<unsigned> idx_A_AC,
                    std::vector<unsigned> idx_A_ABC,
                             dpd_varray_view<const T> B,
                    std::vector<unsigned> idx_B_only,
                    std::vector<unsigned> idx_B_AB,
                    std::vector<unsigned> idx_B_BC,
                    std::vector<unsigned> idx_B_ABC,
                    T beta_, dpd_varray_view<      T> C,
                    std::vector<unsigned> idx_C_only,
                    std::vector<unsigned> idx_C_AC,
                    std::vector<unsigned> idx_C_BC,
                    std::vector<unsigned> idx_C_ABC)
{
    using stl_ext::intersection;
    using stl_ext::exclusion;
    using stl_ext::select_from;
    using stl_ext::permute;

    unsigned nirrep = A.num_irreps();

    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();
    unsigned ndim_C = C.dimension();

    unsigned ndim_A_only = idx_A_only.size();
    unsigned ndim_B_only = idx_B_only.size();
    unsigned ndim_C_only = idx_C_only.size();
    unsigned ndim_AC = idx_C_AC.size();
    unsigned ndim_BC = idx_C_BC.size();
    unsigned ndim_AB = idx_A_AB.size();
    unsigned ndim_ABC = idx_A_ABC.size();

    std::vector<len_type> len_A(ndim_A);
    for (unsigned i = 0;i < ndim_A;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            len_A[i] += A.length(i, irrep);
    }

    std::vector<len_type> len_B(ndim_B);
    for (unsigned i = 0;i < ndim_B;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            len_B[i] += B.length(i, irrep);
    }

    std::vector<len_type> len_C(ndim_C);
    for (unsigned i = 0;i < ndim_C;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            len_C[i] += C.length(i, irrep);
    }

    auto iperm_A = detail::inverse_permutation(A.permutation());
    std::vector<stride_type> stride_A(ndim_A, 1);
    for (unsigned i = 1;i < ndim_A;i++)
    {
        stride_A[iperm_A[i]] = stride_A[iperm_A[i-1]]*len_A[iperm_A[i-1]];
    }

    auto iperm_B = detail::inverse_permutation(B.permutation());
    std::vector<stride_type> stride_B(ndim_B, 1);
    for (unsigned i = 1;i < ndim_B;i++)
    {
        stride_B[iperm_B[i]] = stride_B[iperm_B[i-1]]*len_B[iperm_B[i-1]];
    }

    auto iperm_C = detail::inverse_permutation(C.permutation());
    std::vector<stride_type> stride_C(ndim_C, 1);
    for (unsigned i = 1;i < ndim_C;i++)
    {
        stride_C[iperm_C[i]] = stride_C[iperm_C[i-1]]*len_C[iperm_C[i-1]];
    }

    auto perm_AC = detail::sort_by_stride(select_from(stride_C, idx_C_AC),
                                          select_from(stride_A, idx_A_AC));
    auto perm_BC = detail::sort_by_stride(select_from(stride_C, idx_C_BC),
                                          select_from(stride_B, idx_B_BC));
    auto perm_AB = detail::sort_by_stride(select_from(stride_A, idx_A_AB),
                                          select_from(stride_B, idx_B_AB));

    permute(idx_A_AC, perm_AC);
    permute(idx_A_AB, perm_AB);
    permute(idx_B_AB, perm_AB);
    permute(idx_B_BC, perm_BC);
    permute(idx_C_AC, perm_AC);
    permute(idx_C_BC, perm_BC);

    stride_type dense_A = 1;
    stride_type nblock_A = 1;
    for (unsigned i : idx_A_only)
    {
        dense_A *= len_A[i];
        nblock_A *= nirrep;
    }
    dense_A /= nblock_A;

    stride_type dense_B = 1;
    stride_type nblock_B = 1;
    for (unsigned i : idx_B_only)
    {
        dense_B *= len_B[i];
        nblock_B *= nirrep;
    }
    dense_B /= nblock_B;

    stride_type dense_C = 1;
    stride_type nblock_C = 1;
    for (unsigned i : idx_C_only)
    {
        dense_C *= len_C[i];
        nblock_C *= nirrep;
    }
    dense_C /= nblock_C;

    stride_type dense_AC = 1;
    stride_type nblock_AC = 1;
    for (unsigned i : idx_C_AC)
    {
        dense_AC *= len_A[i];
        nblock_AC *= nirrep;
    }
    dense_AC /= nblock_AC;

    stride_type dense_BC = 1;
    stride_type nblock_BC = 1;
    for (unsigned i : idx_C_BC)
    {
        dense_BC *= len_B[i];
        nblock_BC *= nirrep;
    }
    dense_BC /= nblock_BC;

    stride_type dense_AB = 1;
    stride_type nblock_AB = 1;
    for (unsigned i : idx_A_AB)
    {
        dense_AB *= len_A[i];
        nblock_AB *= nirrep;
    }
    dense_AB /= nblock_AB;

    stride_type dense_ABC = 1;
    stride_type nblock_ABC = 1;
    for (unsigned i : idx_A_ABC)
    {
        dense_ABC *= len_A[i];
        nblock_ABC *= nirrep;
    }
    dense_ABC /= nblock_ABC;

    if (nblock_A > 1) nblock_A /= nirrep;
    if (nblock_B > 1) nblock_B /= nirrep;
    if (nblock_C > 1) nblock_C /= nirrep;
    if (nblock_AC > 1) nblock_AC /= nirrep;
    if (nblock_BC > 1) nblock_BC /= nirrep;
    if (nblock_AB > 1) nblock_AB /= nirrep;
    if (nblock_ABC > 1) nblock_ABC /= nirrep;

    std::vector<unsigned> irreps_A(ndim_A);
    std::vector<unsigned> irreps_B(ndim_B);
    std::vector<unsigned> irreps_C(ndim_C);

    for (unsigned irrep_ABC = 0;irrep_ABC < nirrep;irrep_ABC++)
    {
        for (unsigned irrep_AC = 0;irrep_AC < nirrep;irrep_AC++)
        {
            for (unsigned irrep_BC = 0;irrep_BC < nirrep;irrep_BC++)
            {
                unsigned irrep_C = C.irrep()^irrep_ABC^irrep_AC^irrep_BC;

                if (ndim_ABC == 0 && irrep_ABC != 0) continue;
                if (ndim_AC == 0 && irrep_AC != 0) continue;
                if (ndim_BC == 0 && irrep_BC != 0) continue;
                if (ndim_C_only == 0 && irrep_C != 0) continue;

                for (stride_type block_ABC = 0;block_ABC < nblock_ABC;block_ABC++)
                {
                    detail::assign_irreps(ndim_ABC, irrep_ABC, nirrep, block_ABC,
                                  irreps_A, idx_A_ABC, irreps_B, idx_B_ABC, irreps_C, idx_C_ABC);

                    for (stride_type block_AC = 0;block_AC < nblock_AC;block_AC++)
                    {
                        detail::assign_irreps(ndim_AC, irrep_AC, nirrep, block_AC,
                                      irreps_A, idx_A_AC, irreps_C, idx_C_AC);

                        for (stride_type block_BC = 0;block_BC < nblock_BC;block_BC++)
                        {
                            detail::assign_irreps(ndim_BC, irrep_BC, nirrep, block_BC,
                                          irreps_B, idx_B_BC, irreps_C, idx_C_BC);

                            for (stride_type block_C = 0;block_C < nblock_C;block_C++)
                            {
                                detail::assign_irreps(ndim_C_only, irrep_C, nirrep, block_C,
                                              irreps_C, idx_C_only);

                                auto local_C = C(irreps_C);

                                auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
                                auto len_AC = stl_ext::select_from(local_C.lengths(), idx_C_AC);
                                auto len_BC = stl_ext::select_from(local_C.lengths(), idx_C_BC);
                                auto len_C_only = stl_ext::select_from(local_C.lengths(), idx_C_only);
                                auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);
                                auto stride_C_AC = stl_ext::select_from(local_C.strides(), idx_C_AC);
                                auto stride_C_BC = stl_ext::select_from(local_C.strides(), idx_C_BC);
                                auto stride_C_C = stl_ext::select_from(local_C.strides(), idx_C_only);

                                T beta = beta_;

                                for (unsigned irrep_AB = 0;irrep_AB < nirrep;irrep_AB++)
                                {
                                    unsigned irrep_A = A.irrep()^irrep_ABC^irrep_AB^irrep_AC;
                                    unsigned irrep_B = B.irrep()^irrep_ABC^irrep_AB^irrep_BC;

                                    if (ndim_AB == 0 && irrep_AB != 0) continue;
                                    if (ndim_A_only == 0 && irrep_A != 0) continue;
                                    if (ndim_B_only == 0 && irrep_B != 0) continue;

                                    for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
                                    {
                                        detail::assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
                                                      irreps_A, idx_A_AB, irreps_B, idx_B_AB);

                                        for (stride_type block_A = 0;block_A < nblock_A;block_A++)
                                        {
                                            detail::assign_irreps(ndim_A_only, irrep_A, nirrep, block_A,
                                                          irreps_A, idx_A_only);

                                            auto local_A = A(irreps_A);

                                            auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
                                            auto len_A_only = stl_ext::select_from(local_A.lengths(), idx_A_only);
                                            auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
                                            auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
                                            auto stride_A_AC = stl_ext::select_from(local_A.strides(), idx_A_AC);
                                            auto stride_A_A = stl_ext::select_from(local_A.strides(), idx_A_only);

                                            for (stride_type block_B = 0;block_B < nblock_B;block_B++)
                                            {
                                                detail::assign_irreps(ndim_B_only, irrep_B, nirrep, block_B,
                                                              irreps_B, idx_B_only);

                                                auto local_B = B(irreps_B);

                                                auto len_B_only = stl_ext::select_from(local_B.lengths(), idx_B_only);
                                                auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
                                                auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);
                                                auto stride_B_BC = stl_ext::select_from(local_B.strides(), idx_B_BC);
                                                auto stride_B_B = stl_ext::select_from(local_B.strides(), idx_B_only);

                                                mult(comm, cfg, len_A_only, len_B_only, len_C_only, len_AB, len_AC, len_BC, len_ABC,
                                                     alpha, false, local_A.data(), stride_A_A, stride_A_AB, stride_A_AC, stride_A_ABC,
                                                            false, local_B.data(), stride_B_B, stride_B_AB, stride_B_BC, stride_B_ABC,
                                                      beta, false, local_C.data(), stride_C_C, stride_C_AC, stride_C_BC, stride_C_ABC);
                                            }
                                        }
                                    }
                                }

                                beta = T(1);
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void dpd_mult(const communicator& comm, const config& cfg,
              T alpha, bool conj_A, const dpd_varray_view<const T>& A,
              const std::vector<unsigned>& idx_A_only,
              const std::vector<unsigned>& idx_A_AB,
              const std::vector<unsigned>& idx_A_AC,
              const std::vector<unsigned>& idx_A_ABC,
                       bool conj_B, const dpd_varray_view<const T>& B,
              const std::vector<unsigned>& idx_B_only,
              const std::vector<unsigned>& idx_B_AB,
              const std::vector<unsigned>& idx_B_BC,
              const std::vector<unsigned>& idx_B_ABC,
              T  beta, bool conj_C, const dpd_varray_view<      T>& C,
              const std::vector<unsigned>& idx_C_only,
              const std::vector<unsigned>& idx_C_AC,
              const std::vector<unsigned>& idx_C_BC,
              const std::vector<unsigned>& idx_C_ABC)
{
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C);

    if (impl == REFERENCE)
    {
        dpd_mult_ref(comm, cfg,
                     alpha, A, idx_A_only, idx_A_AB, idx_A_AC, idx_A_ABC,
                            B, idx_B_only, idx_B_AB, idx_B_BC, idx_B_ABC,
                      beta, C, idx_C_only, idx_C_AC, idx_C_BC, idx_C_ABC);
    }
    else if (idx_A_only.empty() && idx_B_only.empty() &&
             idx_C_only.empty() && idx_C_ABC.empty())
    {
        dpd_contract_block(comm, cfg,
                           alpha, A, idx_A_AB, idx_A_AC,
                                  B, idx_B_AB, idx_B_BC,
                            beta, C, idx_C_AC, idx_C_BC);
    }
    else
    {
        dpd_mult_block(comm, cfg,
                       alpha, A, idx_A_only, idx_A_AB, idx_A_AC, idx_A_ABC,
                              B, idx_B_only, idx_B_AB, idx_B_BC, idx_B_ABC,
                        beta, C, idx_C_only, idx_C_AC, idx_C_BC, idx_C_ABC);
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void dpd_mult(const communicator& comm, const config& cfg, \
                       T alpha, bool conj_A, const dpd_varray_view<const T>& A, \
                       const std::vector<unsigned>& idx_A_only, \
                       const std::vector<unsigned>& idx_A_AB, \
                       const std::vector<unsigned>& idx_A_AC, \
                       const std::vector<unsigned>& idx_A_ABC, \
                                bool conj_B, const dpd_varray_view<const T>& B, \
                       const std::vector<unsigned>& idx_B_only, \
                       const std::vector<unsigned>& idx_B_AB, \
                       const std::vector<unsigned>& idx_B_BC, \
                       const std::vector<unsigned>& idx_B_ABC, \
                       T  beta, bool conj_C, const dpd_varray_view<      T>& C, \
                       const std::vector<unsigned>& idx_C_only, \
                       const std::vector<unsigned>& idx_C_AC, \
                       const std::vector<unsigned>& idx_C_BC, \
                       const std::vector<unsigned>& idx_C_ABC);
#include "configs/foreach_type.h"

}
}
