#ifndef _TBLIS_BATCHED_TENSOR_CONTRACT_HPP_
#define _TBLIS_BATCHED_TENSOR_CONTRACT_HPP_

#include "iface/3t/mult.h"

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

#include "tblis_batched_tensor.hpp"

#include "../external/stl_ext/include/iostream.hpp"

extern std::atomic<long> flops;

namespace tblis
{

namespace internal
{

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

}

template <typename T>
int contract_batch_ref(T alpha, const_batched_tensor_view<T> A, const label_type* idx_A,
                                const_batched_tensor_view<T> B, const label_type* idx_B,
                       T  beta,       batched_tensor_view<T> C, const label_type* idx_C)
{
    unsigned ndim_A = A.dense_dimension();
    unsigned ndim_B = B.dense_dimension();
    unsigned ndim_C = C.dense_dimension();

    unsigned batch_ndim_A = A.batched_dimension();
    unsigned batch_ndim_B = B.batched_dimension();
    unsigned batch_ndim_C = C.batched_dimension();

    std::vector<len_type> len_A(A.lengths().begin(), A.lengths().begin()+ndim_A);
    std::vector<len_type> len_B(B.lengths().begin(), B.lengths().begin()+ndim_B);
    std::vector<len_type> len_C(C.lengths().begin(), C.lengths().begin()+ndim_C);

    std::vector<stride_type> stride_A = A.strides();
    std::vector<stride_type> stride_B = B.strides();
    std::vector<stride_type> stride_C = C.strides();

    std::vector<label_type> dense_idx_A(idx_A, idx_A+ndim_A);
    std::vector<label_type> dense_idx_B(idx_B, idx_B+ndim_B);
    std::vector<label_type> dense_idx_C(idx_C, idx_C+ndim_C);

    std::vector<label_type> batch_idx_A(idx_A+ndim_A, idx_A+ndim_A+batch_ndim_A);
    std::vector<label_type> batch_idx_B(idx_B+ndim_B, idx_B+ndim_B+batch_ndim_B);
    std::vector<label_type> batch_idx_C(idx_C+ndim_C, idx_C+ndim_C+batch_ndim_C);

    std::vector<stride_type> batch_stride_A_A(batch_ndim_A);
    std::vector<stride_type> batch_stride_A_B(batch_ndim_A);
    std::vector<stride_type> batch_stride_A_C(batch_ndim_A);

    for (int i = 0;i < batch_ndim_A;i++)
    {
        for (int j = 0;j < ndim_A;)
        {
            if (dense_idx_A[j] == batch_idx_A[i])
            {
                batch_stride_A_A[i] += stride_A[j];
                len_A.erase(len_A.begin()+j);
                stride_A.erase(stride_A.begin()+j);
                dense_idx_A.erase(dense_idx_A.begin()+j);
                ndim_A--;
            }
            else j++;
        }

        for (int j = 0;j < ndim_B;)
        {
            if (dense_idx_B[j] == batch_idx_A[i])
            {
                batch_stride_A_B[i] += stride_B[j];
                len_B.erase(len_B.begin()+j);
                stride_B.erase(stride_B.begin()+j);
                dense_idx_B.erase(dense_idx_B.begin()+j);
                ndim_B--;
            }
            else j++;
        }

        for (int j = 0;j < ndim_C;)
        {
            if (dense_idx_C[j] == batch_idx_A[i])
            {
                batch_stride_A_C[i] += stride_C[j];
                len_C.erase(len_C.begin()+j);
                stride_C.erase(stride_C.begin()+j);
                dense_idx_C.erase(dense_idx_C.begin()+j);
                ndim_C--;
            }
            else j++;
        }
    }

    std::vector<stride_type> batch_stride_B_A(batch_ndim_B);
    std::vector<stride_type> batch_stride_B_B(batch_ndim_B);
    std::vector<stride_type> batch_stride_B_C(batch_ndim_B);

    for (int i = 0;i < batch_ndim_B;i++)
    {
        for (int j = 0;j < ndim_A;)
        {
            if (dense_idx_A[j] == batch_idx_B[i])
            {
                batch_stride_B_A[i] += stride_A[j];
                len_A.erase(len_A.begin()+j);
                stride_A.erase(stride_A.begin()+j);
                dense_idx_A.erase(dense_idx_A.begin()+j);
                ndim_A--;
            }
            else j++;
        }

        for (int j = 0;j < ndim_B;)
        {
            if (dense_idx_B[j] == batch_idx_B[i])
            {
                batch_stride_B_B[i] += stride_B[j];
                len_B.erase(len_B.begin()+j);
                stride_B.erase(stride_B.begin()+j);
                dense_idx_B.erase(dense_idx_B.begin()+j);
                ndim_B--;
            }
            else j++;
        }

        for (int j = 0;j < ndim_C;)
        {
            if (dense_idx_C[j] == batch_idx_B[i])
            {
                batch_stride_B_C[i] += stride_C[j];
                len_C.erase(len_C.begin()+j);
                stride_C.erase(stride_C.begin()+j);
                dense_idx_C.erase(dense_idx_C.begin()+j);
                ndim_C--;
            }
            else j++;
        }
    }

    std::vector<stride_type> batch_stride_C_A(batch_ndim_C);
    std::vector<stride_type> batch_stride_C_B(batch_ndim_C);
    std::vector<stride_type> batch_stride_C_C(batch_ndim_C);

    for (int i = 0;i < batch_ndim_C;i++)
    {
        for (int j = 0;j < ndim_A;)
        {
            if (idx_A[j] == batch_idx_C[i])
            {
                batch_stride_C_A[i] += stride_A[j];
                len_A.erase(len_A.begin()+j);
                stride_A.erase(stride_A.begin()+j);
                dense_idx_A.erase(dense_idx_A.begin()+j);
                ndim_A--;
            }
            else j++;
        }

        for (int j = 0;j < ndim_B;)
        {
            if (idx_B[j] == batch_idx_C[i])
            {
                batch_stride_C_B[i] += stride_B[j];
                len_B.erase(len_B.begin()+j);
                stride_B.erase(stride_B.begin()+j);
                dense_idx_B.erase(dense_idx_B.begin()+j);
                ndim_B--;
            }
            else j++;
        }

        for (int j = 0;j < ndim_C;)
        {
            if (idx_C[j] == batch_idx_C[i])
            {
                batch_stride_C_C[i] += stride_C[j];
                len_C.erase(len_C.begin()+j);
                stride_C.erase(stride_C.begin()+j);
                dense_idx_C.erase(dense_idx_C.begin()+j);
                ndim_C--;
            }
            else j++;
        }
    }

    tblis_tensor tensor_A, tensor_B, tensor_C;

    tensor_A.conj = 0;
    tensor_A.ndim = ndim_A;
    tensor_A.len = len_A.data();
    tensor_A.stride = stride_A.data();
    tensor_A.type = type_tag<T>::value;

    tensor_B.conj = 0;
    tensor_B.ndim = ndim_B;
    tensor_B.len = len_B.data();
    tensor_B.stride = stride_B.data();
    tensor_B.type = type_tag<T>::value;

    tensor_C.conj = 0;
    tensor_C.ndim = ndim_C;
    tensor_C.len = len_C.data();
    tensor_C.stride = stride_C.data();
    tensor_C.type = type_tag<T>::value;

    auto dense_idx_AB = stl_ext::intersection(dense_idx_A, dense_idx_B);
    auto dense_idx_AC = stl_ext::intersection(dense_idx_A, dense_idx_C);
    auto dense_idx_BC = stl_ext::intersection(dense_idx_B, dense_idx_C);
    auto dense_len_AB = stl_ext::select_from(len_A, dense_idx_A, dense_idx_AB);
    auto dense_len_AC = stl_ext::select_from(len_A, dense_idx_A, dense_idx_AC);
    auto dense_len_BC = stl_ext::select_from(len_B, dense_idx_B, dense_idx_BC);

    len_type dense_M = stl_ext::prod(dense_len_AC);
    len_type dense_N = stl_ext::prod(dense_len_BC);

    auto local_flops = 2*stl_ext::prod(dense_len_AB)*dense_M*dense_N;

#define OUTER_THREADING 1

#if OUTER_THREADING
#pragma omp parallel for schedule(static,1), firstprivate(tensor_A, tensor_B, tensor_C)
#endif
    for (unsigned batch_C = 0;batch_C < C.num_batches();batch_C++)
    {
        T local_beta = beta;

        const T* local_C = C.batch_data(batch_C);
        for (unsigned i = 0;i < batch_ndim_C;i++)
        {
            local_C += batch_stride_C_C[i];
        }

        for (unsigned batch_A = 0;batch_A < A.num_batches();batch_A++)
        {

            bool ok = true;
            for (unsigned i = 0;ok && i < batch_ndim_C;i++)
            {
                for (unsigned j = 0;ok && j < batch_ndim_A;j++)
                {
                    if (batch_idx_C[i] == batch_idx_A[j] &&
                        C.batch_indices()[batch_C][i] !=
                        A.batch_indices()[batch_A][j]) ok = false;
                }
            }
            if (!ok) continue;

            const T* local_A = A.batch_data(batch_A);
            for (unsigned i = 0;i < batch_ndim_C;i++)
            {
                local_A += batch_stride_C_A[i];
            }
            for (unsigned i = 0;i < batch_ndim_A;i++)
            {
                local_A += batch_stride_A_A[i];
                local_C += batch_stride_A_C[i];
            }

            for (unsigned batch_B = 0;batch_B < B.num_batches();batch_B++)
            {
                bool ok = true;
                for (unsigned i = 0;ok && i < batch_ndim_C;i++)
                {
                    for (unsigned j = 0;ok && j < batch_ndim_B;j++)
                    {
                        if (batch_idx_C[i] == batch_idx_B[j] &&
                            C.batch_indices()[batch_C][i] !=
                            B.batch_indices()[batch_B][j]) ok = false;
                    }
                }
                for (unsigned i = 0;ok && i < batch_ndim_A;i++)
                {
                    for (unsigned j = 0;ok && j < batch_ndim_B;j++)
                    {
                        if (batch_idx_A[i] == batch_idx_B[j] &&
                            A.batch_indices()[batch_A][i] !=
                            B.batch_indices()[batch_B][j]) ok = false;
                    }
                }
                if (!ok) continue;

                //printf("%ld %ld %ld - %ld %ld %ld - %ld %ld %ld %ld\n",
                //       A.batch_indices()[batch_A][0],
                //       A.batch_indices()[batch_A][1],
                //       A.batch_indices()[batch_A][2],
                //       B.batch_indices()[batch_B][0],
                //       B.batch_indices()[batch_B][1],
                //       B.batch_indices()[batch_B][2],
                //       C.batch_indices()[batch_C][0],
                //       C.batch_indices()[batch_C][1],
                //       C.batch_indices()[batch_C][2],
                //       C.batch_indices()[batch_C][3]);

                const T* local_B = B.batch_data(batch_B);
                for (unsigned i = 0;i < batch_ndim_C;i++)
                {
                    local_B += batch_stride_C_B[i];
                }
                for (unsigned i = 0;i < batch_ndim_A;i++)
                {
                    local_B += batch_stride_A_B[i];
                }
                for (unsigned i = 0;i < batch_ndim_B;i++)
                {
                    local_A += batch_stride_B_A[i];
                    local_B += batch_stride_B_B[i];
                    local_C += batch_stride_B_C[i];
                }

                tensor_A.data = (void*)local_A;
                tensor_B.data = (void*)local_B;
                tensor_C.data = (void*)local_C;

                tensor_A.alpha<T>() = alpha;
                tensor_B.alpha<T>() = T(1);
                tensor_C.alpha<T>() = local_beta;
                local_beta = T(1);

                flops += local_flops;
                //printf("%d %d %d\n", batch_A, batch_B, batch_C);

                tblis_tensor_mult(
#if OUTER_THREADING
                                  tblis_single,
#else
                                  nullptr,
#endif
                                  nullptr,
                                  &tensor_A, dense_idx_A.data(),
                                  &tensor_B, dense_idx_B.data(),
                                  &tensor_C, dense_idx_C.data());
            }
        }
    }
}

template <typename T>
int contract_batch(T alpha, const_batched_tensor_view<T> A, const label_type* idx_A_,
                            const_batched_tensor_view<T> B, const label_type* idx_B_,
                   T  beta,       batched_tensor_view<T> C, const label_type* idx_C_)
{
    using stl_ext::intersection;
    using stl_ext::exclusion;
    using stl_ext::select_from;
    using stl_ext::permuted;

    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();
    unsigned ndim_C = C.dimension();

    unsigned dense_ndim_A = A.dense_dimension();
    unsigned dense_ndim_B = B.dense_dimension();
    unsigned dense_ndim_C = C.dense_dimension();

    unsigned batch_ndim_A = A.batched_dimension();
    unsigned batch_ndim_B = B.batched_dimension();
    unsigned batch_ndim_C = C.batched_dimension();

    std::vector<label_type> idx_A(idx_A_, idx_A_+ndim_A);
    std::vector<label_type> idx_B(idx_B_, idx_B_+ndim_B);
    std::vector<label_type> idx_C(idx_C_, idx_C_+ndim_C);

    std::vector<label_type> dense_idx_A(idx_A_, idx_A_+dense_ndim_A);
    std::vector<label_type> dense_idx_B(idx_B_, idx_B_+dense_ndim_B);
    std::vector<label_type> dense_idx_C(idx_C_, idx_C_+dense_ndim_C);

    std::vector<label_type> batch_idx_A(idx_A_+dense_ndim_A, idx_A_+ndim_A);
    std::vector<label_type> batch_idx_B(idx_B_+dense_ndim_B, idx_B_+ndim_B);
    std::vector<label_type> batch_idx_C(idx_C_+dense_ndim_C, idx_C_+ndim_C);

    const std::vector<len_type>& len_A = A.lengths();
    const std::vector<len_type>& len_B = B.lengths();
    const std::vector<len_type>& len_C = C.lengths();

    std::vector<len_type> dense_len_A(A.lengths().begin(), A.lengths().begin()+dense_ndim_A);
    std::vector<len_type> dense_len_B(B.lengths().begin(), B.lengths().begin()+dense_ndim_B);
    std::vector<len_type> dense_len_C(C.lengths().begin(), C.lengths().begin()+dense_ndim_C);

    std::vector<len_type> batch_len_A(A.lengths().begin()+dense_ndim_A, A.lengths().end());
    std::vector<len_type> batch_len_B(B.lengths().begin()+dense_ndim_B, B.lengths().end());
    std::vector<len_type> batch_len_C(C.lengths().begin()+dense_ndim_C, C.lengths().end());

    std::vector<len_type> stride_A(ndim_A);
    std::vector<len_type> stride_B(ndim_B);
    std::vector<len_type> stride_C(ndim_C);
    std::copy_n(A.strides().begin(), dense_ndim_A, stride_A.begin());
    std::copy_n(B.strides().begin(), dense_ndim_B, stride_B.begin());
    std::copy_n(C.strides().begin(), dense_ndim_C, stride_C.begin());

    std::vector<stride_type> dense_stride_A = A.strides();
    std::vector<stride_type> dense_stride_B = B.strides();
    std::vector<stride_type> dense_stride_C = C.strides();

    auto dense_idx_AB = intersection(dense_idx_A, dense_idx_B);
    auto dense_len_AB = select_from(dense_len_A, dense_idx_A, dense_idx_AB);
    auto dense_stride_A_AB = select_from(dense_stride_A, dense_idx_A, dense_idx_AB);
    auto dense_stride_B_AB = select_from(dense_stride_B, dense_idx_B, dense_idx_AB);

    auto dense_idx_AC = intersection(dense_idx_A, dense_idx_C);
    auto dense_len_AC = select_from(dense_len_A, dense_idx_A, dense_idx_AC);
    auto dense_stride_A_AC = select_from(dense_stride_A, dense_idx_A, dense_idx_AC);
    auto dense_stride_C_AC = select_from(dense_stride_C, dense_idx_C, dense_idx_AC);

    auto dense_idx_BC = intersection(dense_idx_B, dense_idx_C);
    auto dense_len_BC = select_from(dense_len_B, dense_idx_B, dense_idx_BC);
    auto dense_stride_B_BC = select_from(dense_stride_B, dense_idx_B, dense_idx_BC);
    auto dense_stride_C_BC = select_from(dense_stride_C, dense_idx_C, dense_idx_BC);

    auto batch_idx_AB = exclusion(intersection(idx_A, idx_B), dense_idx_AB);
    auto batch_idx_AC = exclusion(intersection(idx_A, idx_C), dense_idx_AC);
    auto batch_idx_BC = exclusion(intersection(idx_B, idx_C), dense_idx_BC);

    auto batch_len_AB = select_from(batch_len_A, batch_idx_A, batch_idx_AB);
    auto batch_len_AC = select_from(batch_len_A, batch_idx_A, batch_idx_AC);
    auto batch_len_BC = select_from(batch_len_B, batch_idx_B, batch_idx_BC);

    std::vector<stride_type> off_stride_AB(batch_idx_AB.size()+batch_idx_AC.size()+batch_idx_BC.size());
    std::vector<stride_type> off_stride_AC(batch_idx_AC.size()+batch_idx_AB.size()+batch_idx_BC.size());
    std::vector<stride_type> off_stride_BC(batch_idx_BC.size()+batch_idx_AB.size()+batch_idx_AC.size());
    if (!batch_idx_AB.empty())
    stl_ext::prefix_sum(batch_len_AB.begin(), batch_len_AB.end(),
                        off_stride_AB.begin(), 1, std::multiplies<stride_type>());
    if (!batch_idx_AC.empty())
    stl_ext::prefix_sum(batch_len_AC.begin(), batch_len_AC.end(),
                        off_stride_AC.begin(), 1, std::multiplies<stride_type>());
    if (!batch_idx_BC.empty())
    stl_ext::prefix_sum(batch_len_BC.begin(), batch_len_BC.end(),
                        off_stride_BC.begin(), 1, std::multiplies<stride_type>());

    len_type batch_M = stl_ext::prod(batch_len_AC);
    len_type batch_N = stl_ext::prod(batch_len_BC);
    len_type batch_K = stl_ext::prod(batch_len_AB);

    auto mixed_len_A_AB = select_from(dense_len_A, dense_idx_A, batch_idx_AB);
    auto mixed_len_B_AB = select_from(dense_len_B, dense_idx_B, batch_idx_AB);
    auto mixed_len_A_AC = select_from(dense_len_A, dense_idx_A, batch_idx_AC);
    auto mixed_len_C_AC = select_from(dense_len_C, dense_idx_C, batch_idx_AC);
    auto mixed_len_B_BC = select_from(dense_len_B, dense_idx_B, batch_idx_BC);
    auto mixed_len_C_BC = select_from(dense_len_C, dense_idx_C, batch_idx_BC);

    auto mixed_stride_A_AB = select_from(dense_stride_A, dense_idx_A, batch_idx_AB);
    auto mixed_stride_B_AB = select_from(dense_stride_B, dense_idx_B, batch_idx_AB);
    auto mixed_stride_A_AC = select_from(dense_stride_A, dense_idx_A, batch_idx_AC);
    auto mixed_stride_C_AC = select_from(dense_stride_C, dense_idx_C, batch_idx_AC);
    auto mixed_stride_B_BC = select_from(dense_stride_B, dense_idx_B, batch_idx_BC);
    auto mixed_stride_C_BC = select_from(dense_stride_C, dense_idx_C, batch_idx_BC);

    auto mixed_off_stride_A_AB = select_from(off_stride_AB, batch_idx_AB, dense_idx_A);
    auto mixed_off_stride_B_AB = select_from(off_stride_AB, batch_idx_AB, dense_idx_B);
    auto mixed_off_stride_A_AC = select_from(off_stride_AC, batch_idx_AC, dense_idx_A);
    auto mixed_off_stride_C_AC = select_from(off_stride_AC, batch_idx_AC, dense_idx_C);
    auto mixed_off_stride_B_BC = select_from(off_stride_BC, batch_idx_BC, dense_idx_B);
    auto mixed_off_stride_C_BC = select_from(off_stride_BC, batch_idx_BC, dense_idx_C);

    auto batch_off_stride_A_AB = select_from(off_stride_AB, batch_idx_AB+batch_idx_AC, batch_idx_A);
    auto batch_off_stride_A_AC = select_from(off_stride_AC, batch_idx_AC+batch_idx_AB, batch_idx_A);
    auto batch_off_stride_B_AB = select_from(off_stride_AB, batch_idx_AB+batch_idx_BC, batch_idx_B);
    auto batch_off_stride_B_BC = select_from(off_stride_BC, batch_idx_BC+batch_idx_AB, batch_idx_B);
    auto batch_off_stride_C_AC = select_from(off_stride_AC, batch_idx_AC+batch_idx_BC, batch_idx_C);
    auto batch_off_stride_C_BC = select_from(off_stride_BC, batch_idx_BC+batch_idx_AC, batch_idx_C);

    //std::cout << off_stride_AB << std::endl;
    //std::cout << batch_off_stride_A_AB << std::endl;
    //std::cout << batch_off_stride_B_AB << std::endl << std::endl;

    //std::cout << off_stride_AC << std::endl;
    //std::cout << batch_off_stride_A_AC << std::endl;
    //std::cout << batch_off_stride_C_AC << std::endl << std::endl;

    //std::cout << off_stride_BC << std::endl;
    //std::cout << batch_off_stride_B_BC << std::endl;
    //std::cout << batch_off_stride_C_BC << std::endl << std::endl;

    matrix<const T*> batch_A({batch_M, batch_K}, nullptr);
    matrix<const T*> batch_B({batch_K, batch_N}, nullptr);
    matrix<      T*> batch_C({batch_M, batch_N}, nullptr);

    MArray::viterator<2> it_A_AB(mixed_len_A_AB, mixed_stride_A_AB, mixed_off_stride_A_AB);
    MArray::viterator<2> it_A_AC(mixed_len_A_AC, mixed_stride_A_AC, mixed_off_stride_A_AC);

    for (len_type b = 0;b < A.num_batches();b++)
    {
        len_type off_M = 0, off_K = 0;
        for (unsigned i = 0;i < batch_ndim_A;i++)
        {
            off_M += A.batch_indices()[b][i]*batch_off_stride_A_AC[i];
            off_K += A.batch_indices()[b][i]*batch_off_stride_A_AB[i];
        }

        //printf("A: %ld %ld %ld - %ld %ld\n",
        //       A.batch_indices()[b][0],
        //       A.batch_indices()[b][1],
        //       A.batch_indices()[b][2], off_M, off_K);

        const T* ptr_A = A.batch_data(b);

        while (it_A_AB.next(ptr_A, off_K))
        {
            while (it_A_AC.next(ptr_A, off_M))
            {
                batch_A[off_M][off_K] = ptr_A;
            }
        }
    }

    MArray::viterator<2> it_B_AB(mixed_len_B_AB, mixed_stride_B_AB, mixed_off_stride_B_AB);
    MArray::viterator<2> it_B_BC(mixed_len_B_BC, mixed_stride_B_BC, mixed_off_stride_B_BC);

    for (len_type b = 0;b < B.num_batches();b++)
    {
        len_type off_K = 0, off_N = 0;
        for (unsigned i = 0;i < batch_ndim_B;i++)
        {
            off_K += B.batch_indices()[b][i]*batch_off_stride_B_AB[i];
            off_N += B.batch_indices()[b][i]*batch_off_stride_B_BC[i];
        }

        //printf("B: %ld %ld %ld - %ld %ld\n",
        //       B.batch_indices()[b][0],
        //       B.batch_indices()[b][1],
        //       B.batch_indices()[b][2], off_K, off_N);

        const T* ptr_B = B.batch_data(b);

        while (it_B_AB.next(ptr_B, off_K))
        {
            while (it_B_BC.next(ptr_B, off_N))
            {
                batch_B[off_K][off_N] = ptr_B;
            }
        }
    }

    MArray::viterator<2> it_C_AC(mixed_len_C_AC, mixed_stride_C_AC, mixed_off_stride_C_AC);
    MArray::viterator<2> it_C_BC(mixed_len_C_BC, mixed_stride_C_BC, mixed_off_stride_C_BC);

    for (len_type b = 0;b < C.num_batches();b++)
    {
        len_type off_M = 0, off_N = 0;
        for (unsigned i = 0;i < batch_ndim_C;i++)
        {
            off_M += C.batch_indices()[b][i]*batch_off_stride_C_AC[i];
            off_N += C.batch_indices()[b][i]*batch_off_stride_C_BC[i];
        }

        //printf("C: %ld %ld %ld %ld - %ld %ld\n",
        //       C.batch_indices()[b][0],
        //       C.batch_indices()[b][1],
        //       C.batch_indices()[b][2],
        //       C.batch_indices()[b][3], off_M, off_N);

        T* ptr_C = C.batch_data(b);

        while (it_C_BC.next(ptr_C, off_N))
        {
            while (it_C_AC.next(ptr_C, off_M))
            {
                batch_C[off_M][off_N] = ptr_C;
            }
        }
    }

    fold(dense_len_AB, dense_idx_AB, dense_stride_A_AB, dense_stride_B_AB);
    fold(dense_len_AC, dense_idx_AC, dense_stride_A_AC, dense_stride_C_AC);
    fold(dense_len_BC, dense_idx_BC, dense_stride_B_BC, dense_stride_C_BC);

    /*
     * TODO: sort indices, pair up, and constraint batch_M, etc.?
     */

    std::vector<len_type> nonzero(std::max(batch_M, batch_N)+1);

    len_type dense_M = stl_ext::prod(dense_len_AC);
    len_type dense_N = stl_ext::prod(dense_len_BC);

    auto local_flops = 2*stl_ext::prod(dense_len_AB)*dense_M*dense_N;

    matrix<T> batch_beta({batch_M, batch_N}, beta);

    parallelize
    (
        [&](const communicator& comm)
        {
            for (len_type k = 0;k < batch_K;k++)
            {
                if (batch_M > batch_N)
                {
                    len_type m_min, m_max;
                    std::tie(m_min, m_max, std::ignore) =
                        comm.distribute_over_threads(batch_M);

                    for (len_type m = m_min;m < m_max;m++)
                    {
                        nonzero[m] = 0;
                        for (len_type n = 0;n < batch_N;n++)
                        {
                            if (batch_A[m][k] && batch_B[k][m] && batch_C[m][n])
                                nonzero[m]++;
                        }
                    }
                }
                else
                {
                    len_type n_min, n_max;
                    std::tie(n_min, n_max, std::ignore) =
                        comm.distribute_over_threads(batch_N);

                    for (len_type n = n_min;n < n_max;n++)
                    {
                        nonzero[n] = 0;
                        for (len_type m = 0;m < batch_M;m++)
                        {
                            if (batch_A[m][k] && batch_B[k][m] && batch_C[m][n])
                                nonzero[n]++;
                        }
                    }
                }

                comm.barrier();

                len_type total = stl_ext::sum(nonzero);

                #define INOUT_RATIO 50000

                int nt_outer, nt_inner;
                std::tie(nt_outer, nt_inner) =
                    partition_2x2(comm.num_threads(), INOUT_RATIO*total, dense_M*dense_N);

                //if (comm.master()) printf("%ld:%ld -> %d:%d\n", INOUT_RATIO*total, dense_M*dense_N, nt_outer, nt_inner);

                communicator subcomm = comm.gang(TCI_EVENLY, nt_outer);

                len_type mn_min, mn_max;
                std::tie(mn_min, mn_max, std::ignore) =
                    subcomm.distribute_over_gangs(total);

                //printf("%d: %ld %ld %ld\n", subcomm.gang_num(), total, mn_min, mn_max);

                if (batch_M > batch_N)
                {
                    len_type cur = 0;
                    for (len_type m = 0;m < batch_M;m++)
                    {
                        if (cur+nonzero[m] <= mn_min)
                        {
                            cur += nonzero[m];
                            continue;
                        }
                        else if (cur >= mn_max) break;

                        for (len_type n = 0;n < batch_N;n++)
                        {
                            if (batch_A[m][k] && batch_B[k][n] && batch_C[m][n])
                            {
                                cur++;
                                if (cur > mn_max) break;
                                if (cur <= mn_min) continue;

                                //printf("%ld %ld %ld\n", m, n, k);

                                if (subcomm.master()) flops += local_flops;

                                internal::contract_blis(subcomm, get_default_config(),
                                      dense_len_AB, dense_len_AC, dense_len_BC,
                                                 alpha, batch_A[m][k],
                                                        dense_stride_A_AB,
                                                        dense_stride_A_AC,
                                                        batch_B[k][n],
                                                        dense_stride_B_AB,
                                                        dense_stride_B_BC,
                                      batch_beta[m][n], batch_C[m][n],
                                                        dense_stride_C_AC,
                                                        dense_stride_C_BC);

                                batch_beta[m][n] = T(1);
                            }
                        }
                    }
                }
                else
                {
                    len_type cur = 0;
                    for (len_type n = 0;n < batch_N;n++)
                    {
                        if (cur+nonzero[n] <= mn_min)
                        {
                            cur += nonzero[n];
                            continue;
                        }
                        else if (cur >= mn_max) break;

                        for (len_type m = 0;m < batch_M;m++)
                        {
                            if (batch_A[m][k] && batch_B[k][n] && batch_C[m][n])
                            {
                                cur++;
                                if (cur > mn_max) break;
                                if (cur <= mn_min) continue;

                                //printf("%ld %ld %ld\n", m, n, k);

                                if (subcomm.master()) flops += local_flops;

                                internal::contract_blis(subcomm, get_default_config(),
                                      dense_len_AB, dense_len_AC, dense_len_BC,
                                                 alpha, batch_A[m][k],
                                                        dense_stride_A_AB,
                                                        dense_stride_A_AC,
                                                        batch_B[k][n],
                                                        dense_stride_B_AB,
                                                        dense_stride_B_BC,
                                      batch_beta[m][n], batch_C[m][n],
                                                        dense_stride_C_AC,
                                                        dense_stride_C_BC);

                                batch_beta[m][n] = T(1);
                            }
                        }
                    }
                }

                comm.barrier();
            }
        },
        tblis_get_num_threads()
    );
}

}

#endif
