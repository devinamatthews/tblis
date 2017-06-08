#ifndef _TBLIS_DPD_TENSOR_CONTRACT_HPP_
#define _TBLIS_DPD_TENSOR_CONTRACT_HPP_

#include "batched_tensor_contract.hpp"
#include <numeric>
#include <functional>
#include "src/external/marray/include/dpd_varray_view.hpp"

namespace tblis
{

template <typename T>
void contract_dpd_ref(T alpha, dpd_varray_view<const T> A, const label_type* idx_A_,
                               dpd_varray_view<const T> B, const label_type* idx_B_,
                      T  beta, dpd_varray_view<      T> C, const label_type* idx_C_)
{
    unsigned nirrep = A.num_irreps();
    TBLIS_ASSERT(B.num_irreps() == nirrep);
    TBLIS_ASSERT(B.num_irreps() == nirrep);
    TBLIS_ASSERT(A.irrep()^B.irrep()^C.irrep() == 0);

    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();
    unsigned ndim_C = C.dimension();

    std::vector<label_type> idx_A(idx_A_, idx_A_+ndim_A);
    std::vector<label_type> idx_B(idx_B_, idx_B_+ndim_B);
    std::vector<label_type> idx_C(idx_C_, idx_C_+ndim_C);

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

    varray<T> A2(len_A);
    varray<T> B2(len_B);
    varray<T> C2(len_C);

    struct dirprd
    {
        unsigned operator() (unsigned x, unsigned y) const { return x^y; }
    };

    viterator<0> it_A(std::vector<unsigned>(ndim_A-1, nirrep));
    while (it_A.next())
    {
        auto irreps_A = it_A.position();
        irreps_A.insert(irreps_A.begin(), std::accumulate(irreps_A.begin(), irreps_A.end(), A.irrep(), dirprd()));

        auto local_A = A(irreps_A);
        varray_view<T> local_A2 = A2;

        for (unsigned i = 0;i < ndim_A;i++)
        {
            local_A2.length(i, local_A.length(i));
            local_A2.shift(i, off_A[i][irreps_A[i]]);
        }

        local_A2 = local_A;
    }

    viterator<0> it_B(std::vector<unsigned>(ndim_B-1, nirrep));
    while (it_B.next())
    {
        auto irreps_B = it_B.position();
        irreps_B.insert(irreps_B.begin(), std::accumulate(irreps_B.begin(), irreps_B.end(), B.irrep(), dirprd()));

        auto local_B = B(irreps_B);
        varray_view<T> local_B2 = B2;

        for (unsigned i = 0;i < ndim_B;i++)
        {
            local_B2.length(i, local_B.length(i));
            local_B2.shift(i, off_B[i][irreps_B[i]]);
        }

        local_B2 = local_B;
    }

    viterator<0> it_C(std::vector<unsigned>(ndim_C-1, nirrep));
    while (it_C.next())
    {
        auto irreps_C = it_C.position();
        irreps_C.insert(irreps_C.begin(), std::accumulate(irreps_C.begin(), irreps_C.end(), C.irrep(), dirprd()));

        auto local_C = C(irreps_C);
        varray_view<T> local_C2 = C2;

        for (unsigned i = 0;i < ndim_C;i++)
        {
            local_C2.length(i, local_C.length(i));
            local_C2.shift(i, off_C[i][irreps_C[i]]);
        }

        local_C2 = local_C;
    }

    auto idx_M = stl_ext::intersection(idx_A, idx_C);
    auto idx_N = stl_ext::intersection(idx_B, idx_C);
    auto idx_K = stl_ext::intersection(idx_A, idx_B);

    auto len_M = stl_ext::select_from(len_C, idx_C, idx_M);
    auto len_N = stl_ext::select_from(len_C, idx_C, idx_N);
    auto len_K = stl_ext::select_from(len_A, idx_A, idx_K);

    flops += 2*stl_ext::prod(len_M)*
               stl_ext::prod(len_N)*
               stl_ext::prod(len_K);

    mult<T>(alpha, A2, idx_A_, B2, idx_B_, beta, C2, idx_C_);

    viterator<0> it_C2(std::vector<unsigned>(ndim_C-1, nirrep));
    while (it_C2.next())
    {
        auto irreps_C = it_C2.position();
        irreps_C.insert(irreps_C.begin(), std::accumulate(irreps_C.begin(), irreps_C.end(), C.irrep(), dirprd()));

        auto local_C = C(irreps_C);
        varray_view<T> local_C2 = C2;

        for (unsigned i = 0;i < ndim_C;i++)
        {
            local_C2.length(i, local_C.length(i));
            local_C2.shift(i, off_C[i][irreps_C[i]]);
        }

        local_C = local_C2;
    }
}

template <typename T>
void contract_dpd(T alpha, dpd_varray_view<const T> A, const label_type* idx_A_,
                           dpd_varray_view<const T> B, const label_type* idx_B_,
                  T beta_, dpd_varray_view<      T> C, const label_type* idx_C_)
{
    using stl_ext::intersection;
    using stl_ext::exclusion;
    using stl_ext::select_from;
    using stl_ext::permute;

    unsigned nirrep = A.num_irreps();
    TBLIS_ASSERT(B.num_irreps() == nirrep);
    TBLIS_ASSERT(C.num_irreps() == nirrep);
    TBLIS_ASSERT(A.irrep()^B.irrep()^C.irrep() == 0);

    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();
    unsigned ndim_C = C.dimension();

    unsigned ndim_M = (ndim_A+ndim_C-ndim_B)/2;
    unsigned ndim_N = (ndim_B+ndim_C-ndim_A)/2;
    unsigned ndim_K = (ndim_A+ndim_B-ndim_C)/2;

    std::vector<label_type> idx_A(idx_A_, idx_A_+ndim_A);
    std::vector<label_type> idx_B(idx_B_, idx_B_+ndim_B);
    std::vector<label_type> idx_C(idx_C_, idx_C_+ndim_C);

    {
        auto iperm_A = detail::inverse_permutation(A.permutation());
        auto iperm_B = detail::inverse_permutation(B.permutation());
        auto iperm_C = detail::inverse_permutation(C.permutation());

        A.permute(iperm_A);
        B.permute(iperm_B);
        C.permute(iperm_C);

        permute(idx_A, iperm_A);
        permute(idx_B, iperm_B);
        permute(idx_C, iperm_C);
    }

    auto idx_M = intersection(idx_A, idx_C);
    auto idx_N = intersection(idx_B, idx_C);
    auto idx_K = intersection(idx_A, idx_B);

    std::vector<stride_type> stride_A(ndim_A, 1);
    for (unsigned i = 1;i < ndim_A;i++)
    {
        len_type len = 0;
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            len += A.length(i, irrep);
        stride_A[i] = stride_A[i-1]*len;
    }

    std::vector<stride_type> stride_B(ndim_B, 1);
    for (unsigned i = 1;i < ndim_B;i++)
    {
        len_type len = 0;
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            len += B.length(i, irrep);
        stride_B[i] = stride_B[i-1]*len;
    }

    std::vector<stride_type> stride_C(ndim_C, 1);
    for (unsigned i = 1;i < ndim_C;i++)
    {
        len_type len = 0;
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            len += C.length(i, irrep);
        stride_C[i] = stride_C[i-1]*len;
    }

    {
        auto stride_A_M = select_from(stride_A, idx_A, idx_M);
        auto stride_A_K = select_from(stride_A, idx_A, idx_K);
        auto stride_B_K = select_from(stride_B, idx_B, idx_K);
        auto stride_B_N = select_from(stride_B, idx_B, idx_N);
        auto stride_C_M = select_from(stride_C, idx_C, idx_M);
        auto stride_C_N = select_from(stride_C, idx_C, idx_N);

        permute(idx_M, detail::sort_by_stride(stride_C_M, stride_A_M));
        permute(idx_N, detail::sort_by_stride(stride_C_N, stride_B_N));
        permute(idx_K, detail::sort_by_stride(stride_A_K, stride_B_K));
    }

    std::vector<unsigned> perm_A = range(ndim_A);
    std::vector<unsigned> perm_B = range(ndim_B);
    std::vector<unsigned> perm_C = range(ndim_C);

    auto perm_A_M = select_from(perm_A, idx_A, idx_M);
    auto perm_A_K = select_from(perm_A, idx_A, idx_K);
    auto perm_B_K = select_from(perm_B, idx_B, idx_K);
    auto perm_B_N = select_from(perm_B, idx_B, idx_N);
    auto perm_C_M = select_from(perm_C, idx_C, idx_M);
    auto perm_C_N = select_from(perm_C, idx_C, idx_N);

    stride_type dense_M = 1;
    stride_type block_M = 1;
    for (unsigned i = 0;i < ndim_M;i++)
    {
        len_type local_len = 0;
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            TBLIS_ASSERT(A.length(perm_A_M[i], irrep) ==
                         C.length(perm_C_M[i], irrep));
            local_len += A.length(perm_A_M[i], irrep);
        }
        dense_M *= local_len;
        block_M *= nirrep;
    }
    dense_M /= block_M;
    block_M /= nirrep;

    stride_type dense_N = 1;
    stride_type block_N = 1;
    for (unsigned i = 0;i < ndim_N;i++)
    {
        len_type local_len = 0;
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            TBLIS_ASSERT(B.length(perm_B_N[i], irrep) ==
                         C.length(perm_C_N[i], irrep));
            local_len += B.length(perm_B_N[i], irrep);
        }
        dense_N *= local_len;
        block_N *= nirrep;
    }
    dense_N /= block_N;
    block_N /= nirrep;

    stride_type dense_K = 1;
    stride_type block_K = 1;
    for (unsigned i = 0;i < ndim_K;i++)
    {
        len_type local_len = 0;
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            TBLIS_ASSERT(A.length(perm_A_K[i], irrep) ==
                         B.length(perm_B_K[i], irrep));
            local_len += A.length(perm_A_K[i], irrep);
        }
        dense_K *= local_len;
        block_K *= nirrep;
    }
    dense_K /= block_K;
    block_K /= nirrep;

    std::vector<slot<int, -1>> slot(nirrep*block_M*block_N);

    const config& cfg = get_default_config();
    const bool row_major = cfg.gemm_row_major.value<T>();
    const bool transpose = row_major ? stride_C[perm_C_M[0]] == 1
                                     : stride_C[perm_C_N[0]] == 1;

    if (transpose)
    {
        using std::swap;
        swap(dense_M, dense_N);
        swap(block_M, block_N);
        swap(ndim_M, ndim_N);
        swap(ndim_A, ndim_B);
        swap(idx_A, idx_B);
        swap(idx_M, idx_N);
        swap(stride_A, stride_B);
        swap(perm_A_M, perm_B_N);
        swap(perm_A_K, perm_B_K);
        swap(perm_C_M, perm_C_N);
        swap(A, B);
    }

    unsigned mask = nirrep-1;
    unsigned shift = (nirrep>1) + (nirrep>2) + (nirrep>4);

    auto assign_irreps =
    [&](unsigned ndim, unsigned irrep, stride_type block,
        std::vector<unsigned>& irreps_A, const std::vector<unsigned>& perm_A,
        std::vector<unsigned>& irreps_B, const std::vector<unsigned>& perm_B)
    {
        unsigned irrep0 = irrep;
        for (unsigned i = 1;i < ndim;i++)
        {
            irrep0 ^= irreps_A[perm_A[i]] =
                      irreps_B[perm_B[i]] = block & mask;
            block >>= shift;
        }
        irreps_A[perm_A[0]] = irreps_B[perm_B[0]] = irrep0;
    };

    parallelize([&](const communicator& comm)
    {
        int nt_outer, nt_inner;
        std::tie(nt_outer, nt_inner) =
            partition_2x2(comm.num_threads(),
                          inout_ratio*nirrep*block_M*block_N,
                          dense_M*dense_N);

        communicator subcomm = comm.gang(TCI_EVENLY, nt_outer);
        unsigned gid = subcomm.gang_num();

        std::vector<unsigned> irreps_A(ndim_A);
        std::vector<unsigned> irreps_B(ndim_B);
        std::vector<unsigned> irreps_C(ndim_C);

        stride_type block_idx = 0;
        for (unsigned irrep_K = 0;irrep_K < nirrep;irrep_K++)
        {
            unsigned irrep_M = A.irrep()^irrep_K;
            unsigned irrep_N = B.irrep()^irrep_K;

            for (stride_type iM = 0;iM < block_M;iM++)
            {
                assign_irreps(ndim_M, irrep_M, iM,
                              irreps_A, perm_A_M, irreps_C, perm_C_M);

                for (stride_type iN = 0;iN < block_N;iN++)
                {
                    assign_irreps(ndim_N, irrep_N, iN,
                                  irreps_B, perm_B_N, irreps_C, perm_C_N);

                    if (!slot[block_idx++].try_fill(gid)) continue;

                    tensor_matrix<T> ct(C(irreps_C), perm_C_M, perm_C_N);
                    T beta = beta_;

                    for (stride_type iK = 0;iK < block_K;iK++)
                    {
                        assign_irreps(ndim_K, irrep_K, iK,
                                      irreps_A, perm_A_K, irreps_B, perm_B_K);

                        tensor_matrix<T> at(A(irreps_A), perm_A_M, perm_A_K);
                        tensor_matrix<T> bt(B(irreps_B), perm_B_K, perm_B_N);

                        if (subcomm.master())
                            flops += 2*ct.length(0)*ct.length(1)*at.length(1);

                        internal::TensorGEMM gemm;

                        auto tc = make_gemm_thread_config<T>(cfg, subcomm.num_threads(),
                                                             ct.length(0), ct.length(1), at.length(1));

                        step<0>(gemm).distribute = tc.jc_nt;
                        step<4>(gemm).distribute = tc.ic_nt;
                        step<8>(gemm).distribute = tc.jr_nt;
                        step<9>(gemm).distribute = tc.ir_nt;

                        gemm(subcomm, cfg, alpha, at, bt, beta, ct);

                        beta = T(1);
                    }
                }
            }
        }
    },
    tblis_get_num_threads());
}

}

#endif
