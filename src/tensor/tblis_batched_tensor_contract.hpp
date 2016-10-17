#ifndef _TBLIS_BATCHED_TENSOR_CONTRACT_HPP_
#define _TBLIS_BATCHED_TENSOR_CONTRACT_HPP_

#include "../tblis.h"

namespace tblis
{

template <typename T, typename Config=TBLIS_DEFAULT_CONFIG>
int contract_int(const std::vector<len_type>& len_M,
                 const std::vector<len_type>& len_N,
                 const std::vector<len_type>& len_K,
                 T alpha, const const_matrix_view<const T*>& A,
                          const std::vector<stride_type>& stride_M_A,
                          const std::vector<stride_type>& stride_K_A,
                          const const_matrix_view<const T*>& B,
                          const std::vector<stride_type>& stride_K_B,
                          const std::vector<stride_type>& stride_N_B,
                 T  beta, const const_matrix_view<      T*>& C,
                          const std::vector<stride_type>& stride_M_C,
                          const std::vector<stride_type>& stride_N_C)
{
    len_type batch_len_M = C.length(0);
    len_type batch_len_N = C.length(1);
    len_type batch_len_K = A.length(1);

    std::vector<len_type> nonzero(std::max(batch_len_M, batch_len_N)+1);

    parallelize
    (
        [&](thread_communicator& comm)
        {
            tensor_matrix<T> at(len_M, len_K, nullptr, stride_M_A, stride_K_A);
            tensor_matrix<T> bt(len_K, len_N, nullptr, stride_K_B, stride_N_B);
            tensor_matrix<T> ct(len_M, len_N, nullptr, stride_M_C, stride_N_C);

            len_type dense_len_M = ct.length(0);
            len_type dense_len_N = ct.length(1);
            len_type dense_len_K = at.length(1);

            for (len_type k = 0;k < batch_len_K;k++)
            {
                if (batch_len_M > batch_len_N)
                {
                    len_type m_min, m_max;
                    std::tie(m_min, m_max, std::ignore) =
                        comm.distribute_over_threads(batch_len_M);

                    for (len_type m = m_min;m < m_max;m++)
                    {
                        nonzero[m] = 0;
                        for (len_type n = 0;n < batch_len_N;n++)
                        {
                            if (A[m][k] && B[k][m] && C[m][n]) nonzero[m]++;
                        }
                    }
                }
                else
                {
                    len_type n_min, n_max;
                    std::tie(n_min, n_max, std::ignore) =
                        comm.distribute_over_threads(batch_len_N);

                    for (len_type n = n_min;n < n_max;n++)
                    {
                        nonzero[n] = 0;
                        for (len_type m = 0;m < batch_len_M;m++)
                        {
                            if (A[m][k] && B[k][m] && C[m][n]) nonzero[n]++;
                        }
                    }
                }

                len_type total = stl_ext::sum(nonzero);

                int nt_outer, nt_inner;
                partition_2x2(comm.num_threads(), total,
                              dense_len_M*dense_len_N, nt_outer, nt_inner);

                thread_communicator subcomm = comm.gang_evenly(nt_outer);

                len_type mn_min, mn_max;
                std::tie(mn_min, mn_max, std::ignore) =
                subcomm.distribute_over_gangs(nt_outer, total);

                if (batch_len_M > batch_len_N)
                {
                    len_type cur = 0;
                    for (len_type m = 0;m < batch_len_M;m++)
                    {
                        if (cur+nonzero[m]-1 < mn_min)
                        {
                            cur += nonzero[m];
                            continue;
                        }
                        else if (cur >= mn_max) break;

                        for (len_type n = 0;n < batch_len_N;n++)
                        {
                            if (A[m][k] && B[k][m] && C[m][n])
                            {
                                cur++;
                                if (cur >= mn_max+1) break;
                                if (cur < mn_min+1) continue;

                                at.data(const_cast<T*>(A[m][k]));
                                bt.data(const_cast<T*>(B[k][n]));
                                ct.data(               C[m][n] );

                                TensorGEMM<T, Config>()(subcomm, alpha, at, bt, beta, ct);
                            }
                        }
                    }
                }
                else
                {
                    len_type cur = 0;
                    for (len_type n = 0;n < batch_len_N;n++)
                    {
                        if (cur+nonzero[n]-1 < mn_min)
                        {
                            cur += nonzero[n];
                            continue;
                        }
                        else if (cur >= mn_max) break;

                        for (len_type m = 0;m < batch_len_M;m++)
                        {
                            if (A[m][k] && B[k][m] && C[m][n])
                            {
                                cur++;
                                if (cur >= mn_max+1) break;
                                if (cur < mn_min+1) continue;

                                at.data(const_cast<T*>(A[m][k]));
                                bt.data(const_cast<T*>(B[k][n]));
                                ct.data(               C[m][n] );

                                TensorGEMM<T, Config>()(subcomm, alpha, at, bt, beta, ct);
                            }
                        }
                    }
                }
            }
        }
    );
}

template <typename T>
void contract(T alpha, const_batched_tensor_view<T> A, const std::string& idx_A,
                       const_batched_tensor_view<T> B, const std::string& idx_B,
              T  beta,       batched_tensor_view<T> C, const std::string& idx_C)
{
    std::string dense_idx_A(idx_A.begin(), idx_A.begin()+A.dense_dimension());
    std::string dense_idx_B(idx_B.begin(), idx_B.begin()+B.dense_dimension());
    std::string dense_idx_C(idx_C.begin(), idx_C.begin()+C.dense_dimension());
    std::string batch_idx_A(idx_A.begin()+A.dense_dimension(), idx_A.end());
    std::string batch_idx_B(idx_B.begin()+B.dense_dimension(), idx_B.end());
    std::string batch_idx_C(idx_C.begin()+C.dense_dimension(), idx_C.end());

    unsigned ndim_A = A.dense_dimension();
    unsigned ndim_B = B.dense_dimension();
    unsigned ndim_C = C.dense_dimension();
    std::vector<len_type> len_A(A.lengths().begin(), A.lengths().begin()+ndim_A);
    std::vector<len_type> len_B(B.lengths().begin(), B.lengths().begin()+ndim_B);
    std::vector<len_type> len_C(C.lengths().begin(), C.lengths().begin()+ndim_C);
    std::vector<stride_type> stride_A(A.strides());
    std::vector<stride_type> stride_B(B.strides());
    std::vector<stride_type> stride_C(C.strides());

    diagonal(ndim_A, len_A, stride_A, dense_idx_A);
    diagonal(ndim_B, len_B, stride_B, dense_idx_B);
    diagonal(ndim_C, len_C, stride_C, dense_idx_C);

    using stl_ext::intersection;
    using stl_ext::exclusion;
    using stl_ext::select_from;

    auto idx_AB = intersection(dense_idx_A, dense_idx_B);
    auto len_AB = select_from(len_A, dense_idx_A, idx_AB);
    auto stride_A_AB = select_from(stride_A, dense_idx_A, idx_AB);
    auto stride_B_AB = select_from(stride_B, dense_idx_B, idx_AB);

    auto idx_AC = intersection(dense_idx_A, dense_idx_C);
    auto len_AC = select_from(len_A, dense_idx_A, idx_AC);
    auto stride_A_AC = select_from(stride_A, dense_idx_A, idx_AC);
    auto stride_C_AC = select_from(stride_C, dense_idx_C, idx_AC);

    auto idx_BC = intersection(dense_idx_B, dense_idx_C);
    auto len_BC = select_from(len_B, dense_idx_B, idx_BC);
    auto stride_B_BC = select_from(stride_B, dense_idx_B, idx_BC);
    auto stride_C_BC = select_from(stride_C, dense_idx_C, idx_BC);

    fold(len_AB, {&stride_A_AB, &stride_B_AB}, idx_AB);
    fold(len_AC, {&stride_A_AC, &stride_C_AC}, idx_AC);
    fold(len_BC, {&stride_B_BC, &stride_C_BC}, idx_BC);

    unsigned batch_ndim_A = A.batch_dimension();
    unsigned batch_ndim_B = B.batch_dimension();
    unsigned batch_ndim_C = C.batch_dimension();
    std::vector<len_type> batch_len_A(A.lengths().begin()+ndim_A, A.lengths().end());
    std::vector<len_type> batch_len_B(B.lengths().begin()+ndim_B, B.lengths().end());
    std::vector<len_type> batch_len_C(C.lengths().begin()+ndim_C, C.lengths().end());

    auto batch_idx_AB = intersection(batch_idx_A, batch_idx_B);
    auto batch_idx_AC = intersection(batch_idx_A, batch_idx_C);
    auto batch_idx_BC = intersection(batch_idx_B, batch_idx_C);
    auto batch_len_AB = select_from(batch_len_A, batch_idx_A, batch_idx_AB);
    auto batch_len_AC = select_from(batch_len_A, batch_idx_A, batch_idx_AC);
    auto batch_len_BC = select_from(batch_len_B, batch_idx_B, batch_idx_BC);

    std::vector<stride_type> stride_AB(batch_idx_AB.size(), 1);
    std::vector<stride_type> stride_AC(batch_idx_AC.size(), 1);
    std::vector<stride_type> stride_BC(batch_idx_BC.size(), 1);
    if (!batch_idx_AB.empty())
    std::partial_sum(std::next(batch_len_AB.begin()), batch_len_AB.end(),
                     std::next(stride_AB.begin()), std::multiplies<stride_type>());
    if (!batch_idx_AC.empty())
    std::partial_sum(std::next(batch_len_AC.begin()), batch_len_AC.end(),
                     std::next(stride_AC.begin()), std::multiplies<stride_type>());
    if (!batch_idx_BC.empty())
    std::partial_sum(std::next(batch_len_BC.begin()), batch_len_BC.end(),
                     std::next(stride_BC.begin()), std::multiplies<stride_type>());

    std::vector<stride_type> batch_stride_A_AB(batch_ndim_A);
    std::vector<stride_type> batch_stride_B_AB(batch_ndim_B);
    std::vector<stride_type> batch_stride_A_AC(batch_ndim_A);
    std::vector<stride_type> batch_stride_C_AC(batch_ndim_C);
    std::vector<stride_type> batch_stride_B_BC(batch_ndim_B);
    std::vector<stride_type> batch_stride_C_BC(batch_ndim_C);

    for (unsigned i = 0, i_AB = 0, i_AC = 0;i < batch_ndim_A;i++)
    {
        auto loc = batch_idx_AB.find(batch_idx_A[i]);
        if (loc != std::string::npos)
        {
            batch_stride_A_AB[i_AB++] = stride_AB[loc];
        }
        else
        {
            loc = batch_idx_AC.find(batch_idx_A[i]);
            batch_stride_A_AC[i_AC++] = stride_AC[loc];
        }
    }

    for (unsigned i = 0, i_AB = 0, i_BC = 0;i < batch_ndim_B;i++)
    {
        auto loc = batch_idx_AB.find(batch_idx_B[i]);
        if (loc != std::string::npos)
        {
            batch_stride_B_AB[i_AB++] = stride_AB[loc];
        }
        else
        {
            loc = batch_idx_BC.find(batch_idx_B[i]);
            batch_stride_B_BC[i_BC++] = stride_BC[loc];
        }
    }

    for (unsigned i = 0, i_AC = 0, i_BC = 0;i < batch_ndim_C;i++)
    {
        auto loc = batch_idx_AC.find(batch_idx_C[i]);
        if (loc != std::string::npos)
        {
            batch_stride_C_AC[i_AC++] = stride_AC[loc];
        }
        else
        {
            loc = batch_idx_BC.find(batch_idx_C[i]);
            batch_stride_C_BC[i_BC++] = stride_BC[loc];
        }
    }

    len_type batch_M = (batch_len_AC.empty() ? 1 : batch_len_AC.back()*stride_AC.back());
    len_type batch_N = (batch_len_BC.empty() ? 1 : batch_len_BC.back()*stride_BC.back());
    len_type batch_K = (batch_len_AB.empty() ? 1 : batch_len_AB.back()*stride_AB.back());

    matrix<const T*> batch_A({batch_M, batch_K}, nullptr);
    matrix<const T*> batch_B({batch_K, batch_N}, nullptr);
    matrix<      T*> batch_C({batch_M, batch_N}, nullptr);

    auto indices_A = A.batch_indices();
    auto indices_B = B.batch_indices();
    auto indices_C = C.batch_indices();

    for (len_type b = 0;b < A.num_batches();b++)
    {
        len_type off_M = 1, off_K = 1;
        for (unsigned i = 0;i < batch_ndim_A;i++)
        {
            off_M += indices_A[b][i]*batch_stride_A_AC[i];
            off_K += indices_A[b][i]*batch_stride_A_AB[i];
        }
        batch_A[off_M][off_K] = A.batch_data(b);
    }

    for (len_type b = 0;b < B.num_batches();b++)
    {
        len_type off_K = 1, off_N = 1;
        for (unsigned i = 0;i < batch_ndim_B;i++)
        {
            off_K += indices_B[b][i]*batch_stride_B_AB[i];
            off_N += indices_B[b][i]*batch_stride_B_BC[i];
        }
        batch_B[off_K][off_N] = B.batch_data(b);
    }

    for (len_type b = 0;b < C.num_batches();b++)
    {
        len_type off_M = 1, off_N = 1;
        for (unsigned i = 0;i < batch_ndim_C;i++)
        {
            off_M += indices_C[b][i]*batch_stride_C_AC[i];
            off_N += indices_C[b][i]*batch_stride_C_AC[i];
        }
        batch_C[off_M][off_N] = C.batch_data(b);
    }

    /*
     * TODO: sort indices, pair up, and constraint batch_M, etc.?
     */

    std::vector<unsigned> reorder_AC = MArray::range<unsigned>(idx_AC.size());
    stl_ext::sort(reorder_AC,
    [&](unsigned a, unsigned b)
    {
        return stride_C_AC[a] == stride_C_AC[b] ?
               stride_A_AC[a]  < stride_A_AC[b] :
               stride_C_AC[a]  < stride_C_AC[b];
    });

    std::vector<unsigned> reorder_BC = MArray::range<unsigned>(idx_BC.size());
    stl_ext::sort(reorder_BC,
    [&](unsigned a, unsigned b)
    {
        return stride_C_BC[a] == stride_C_BC[b] ?
               stride_B_BC[a]  < stride_B_BC[b] :
               stride_C_BC[a]  < stride_C_BC[b];
    });

    std::vector<unsigned> reorder_AB = MArray::range<unsigned>(idx_AB.size());
    stl_ext::sort(reorder_AB,
    [&](unsigned a, unsigned b)
    {
        return stride_A_AB[a] == stride_A_AB[b] ?
               stride_B_AB[a]  < stride_B_AB[b] :
               stride_A_AB[a]  < stride_A_AB[b];
    });

    using stl_ext::permuted;

    contract_int<T>(permuted(len_AC, reorder_AC), permuted(len_BC, reorder_BC), permuted(len_AB, reorder_AB),
                    alpha, batch_A, permuted(stride_A_AC, reorder_AC), permuted(stride_A_AB, reorder_AB),
                           batch_B, permuted(stride_B_AB, reorder_AB), permuted(stride_B_BC, reorder_BC),
                     beta, batch_C, permuted(stride_C_AC, reorder_AC), permuted(stride_C_BC, reorder_BC));
}

}

#endif
