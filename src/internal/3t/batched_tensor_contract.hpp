#ifndef _TBLIS_BATCHED_TENSOR_CONTRACT_HPP_
#define _TBLIS_BATCHED_TENSOR_CONTRACT_HPP_

#include "iface/3t/mult.h"

#include "nodes/matrify.hpp"
#include "nodes/partm.hpp"
#include "nodes/gemm_ukr.hpp"

#include "matrix/scatter_tensor_matrix.hpp"

#include "util/gemm_thread.hpp"
#include "util/tensor.hpp"
#include "util/thread.h"
#include "util/basic_types.h"

#include "configs/configs.hpp"

#include "src/external/stl_ext/include/iostream.hpp"

#include <atomic>

namespace tblis
{

extern std::atomic<long> flops;
extern len_type inout_ratio;
extern int outer_threading;
int outer_threading = 1;

template <typename T, T Empty=T()>
class slot
{
    public:
        slot(T value=Empty) : _value(value) {}

        slot(const slot& other) : _value(other._value.load()) {}

        T value() const { return _value.load(); }

        bool is_filled() const
        {
            return value() != Empty;
        }

        bool try_fill(T desired)
        {
            return _try_fill(desired, false);
        }

        void fill(T desired)
        {
            while (!_try_fill(desired, true)) continue;
        }

        void clear()
        {
            _value.store(Empty);
        }

    protected:
        bool _try_fill(T desired, bool weak)
        {
            T expected = Empty;

            if (weak)
            {
                return _value.compare_exchange_weak(expected, desired) ||
                       expected == desired;
            }
            else
            {
                return _value.compare_exchange_strong(expected, desired) ||
                       expected == desired;
            }
        }

        std::atomic<T> _value;
};

namespace internal
{

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

extern MemoryPool BuffersForA, BuffersForB, BuffersForScatter;
extern MemoryPool BuffersForScatter;

using TensorGEMM = partition_gemm_nc<
                     partition_gemm_kc<
                       matrify_and_pack_b<BuffersForB,
                         partition_gemm_mc<
                           matrify_and_pack_a<BuffersForA,
                             matrify_c<BuffersForScatter,
                               partition_gemm_nr<
                                 partition_gemm_mr<
                                   gemm_micro_kernel>>>>>>>>;

}

template <typename T>
void contract_batch_dumb(T alpha, indexed_varray_view<const T> A, const label_type* idx_A,
                                  indexed_varray_view<const T> B, const label_type* idx_B,
                         T  beta,       indexed_varray_view<T> C, const label_type* idx_C)
{
    varray<T> at(A.lengths());
    varray<T> bt(B.lengths());
    varray<T> ct(C.lengths());

    unsigned dense_ndim_A = A.dense_dimension();
    unsigned dense_ndim_B = B.dense_dimension();
    unsigned dense_ndim_C = C.dense_dimension();

    unsigned batched_ndim_A = A.indexed_dimension();
    unsigned batched_ndim_B = B.indexed_dimension();
    unsigned batched_ndim_C = C.indexed_dimension();

    const auto& dense_len_A = A.dense_lengths();
    const auto& dense_len_B = B.dense_lengths();
    const auto& dense_len_C = C.dense_lengths();

    const auto& dense_stride_A = A.dense_strides();
    const auto& dense_stride_B = B.dense_strides();
    const auto& dense_stride_C = C.dense_strides();

    stride_vector packed_stride_A(at.strides().begin(), at.strides().begin()+dense_ndim_A);
    stride_vector packed_stride_B(bt.strides().begin(), bt.strides().begin()+dense_ndim_B);
    stride_vector packed_stride_C(ct.strides().begin(), ct.strides().begin()+dense_ndim_C);

    for (len_type i = 0;i < A.num_indices();i++)
    {
        const T* from = A.data(i);
        T* to = at.data();
        for (len_type j = 0;j < batched_ndim_A;j++)
            to += A.indices()[i][j]*at.stride(j+dense_ndim_A);

        viterator<2> it(dense_len_A, dense_stride_A, packed_stride_A);
        while (it.next(from, to)) *to = *from;
    }

    for (len_type i = 0;i < B.num_indices();i++)
    {
        const T* from = B.data(i);
        T* to = bt.data();
        for (len_type j = 0;j < batched_ndim_B;j++)
            to += B.indices()[i][j]*bt.stride(j+dense_ndim_B);

        viterator<2> it(dense_len_B, dense_stride_B, packed_stride_B);
        while (it.next(from, to)) *to = *from;
    }

    for (len_type i = 0;i < C.num_indices();i++)
    {
        const T* from = C.data(i);
        T* to = ct.data();
        for (len_type j = 0;j < batched_ndim_C;j++)
            to += C.indices()[i][j]*ct.stride(j+dense_ndim_C);

        viterator<2> it(dense_len_C, dense_stride_C, packed_stride_C);
        while (it.next(from, to)) *to = *from;
    }

    mult<T>(alpha, at, idx_A,
                   bt, idx_B,
             beta, ct, idx_C);

    for (len_type i = 0;i < C.num_indices();i++)
    {
        T* to = C.data(i);
        const T* from = ct.data();
        for (len_type j = 0;j < batched_ndim_C;j++)
            from += C.indices()[i][j]*ct.stride(j+dense_ndim_C);

        viterator<2> it(dense_len_C, dense_stride_C, packed_stride_C);
        while (it.next(to, from)) *to = *from;
    }
}

template <typename T>
void contract_batch_ref(T alpha, indexed_varray_view<const T> A, const label_type* idx_A,
                                 indexed_varray_view<const T> B, const label_type* idx_B,
                        T  beta,       indexed_varray_view<T> C, const label_type* idx_C)
{
    unsigned ndim_A = A.dense_dimension();
    unsigned ndim_B = B.dense_dimension();
    unsigned ndim_C = C.dense_dimension();

    unsigned batch_ndim_A = A.indexed_dimension();
    unsigned batch_ndim_B = B.indexed_dimension();
    unsigned batch_ndim_C = C.indexed_dimension();

    auto len_A = A.lengths();
    auto len_B = B.lengths();
    auto len_C = C.lengths();

    auto stride_A = A.dense_strides();
    auto stride_B = B.dense_strides();
    auto stride_C = C.dense_strides();

    label_vector dense_idx_A(idx_A, idx_A+ndim_A);
    label_vector dense_idx_B(idx_B, idx_B+ndim_B);
    label_vector dense_idx_C(idx_C, idx_C+ndim_C);

    label_vector batch_idx_A(idx_A+ndim_A, idx_A+ndim_A+batch_ndim_A);
    label_vector batch_idx_B(idx_B+ndim_B, idx_B+ndim_B+batch_ndim_B);
    label_vector batch_idx_C(idx_C+ndim_C, idx_C+ndim_C+batch_ndim_C);

    stride_vector batch_stride_A_A(batch_ndim_A);
    stride_vector batch_stride_A_B(batch_ndim_A);
    stride_vector batch_stride_A_C(batch_ndim_A);

    for (int i = 0;i < batch_ndim_A;i++)
    {
        for (int j = 0;j < ndim_A;)
        {
            if (dense_idx_A[j] == batch_idx_A[i])
            {
                //printf("A->A: %d(%c) %d(%c)\n", i, batch_idx_A[i], j, dense_idx_A[j]);
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
                //printf("A->B: %d(%c) %d(%c)\n", i, batch_idx_A[i], j, dense_idx_B[j]);
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
                //printf("A->C: %d(%c) %d(%c)\n", i, batch_idx_A[i], j, dense_idx_C[j]);
                batch_stride_A_C[i] += stride_C[j];
                len_C.erase(len_C.begin()+j);
                stride_C.erase(stride_C.begin()+j);
                dense_idx_C.erase(dense_idx_C.begin()+j);
                ndim_C--;
            }
            else j++;
        }
    }

    stride_vector batch_stride_B_A(batch_ndim_B);
    stride_vector batch_stride_B_B(batch_ndim_B);
    stride_vector batch_stride_B_C(batch_ndim_B);

    for (int i = 0;i < batch_ndim_B;i++)
    {
        for (int j = 0;j < ndim_A;)
        {
            if (dense_idx_A[j] == batch_idx_B[i])
            {
                //printf("B->A: %d(%c) %d(%c)\n", i, batch_idx_B[i], j, dense_idx_A[j]);
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
                //printf("B->B: %d(%c) %d(%c)\n", i, batch_idx_B[i], j, dense_idx_B[j]);
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
                //printf("B->C: %d(%c) %d(%c)\n", i, batch_idx_B[i], j, dense_idx_C[j]);
                batch_stride_B_C[i] += stride_C[j];
                len_C.erase(len_C.begin()+j);
                stride_C.erase(stride_C.begin()+j);
                dense_idx_C.erase(dense_idx_C.begin()+j);
                ndim_C--;
            }
            else j++;
        }
    }

    stride_vector batch_stride_C_A(batch_ndim_C);
    stride_vector batch_stride_C_B(batch_ndim_C);
    stride_vector batch_stride_C_C(batch_ndim_C);

    for (int i = 0;i < batch_ndim_C;i++)
    {
        for (int j = 0;j < ndim_A;)
        {
            if (dense_idx_A[j] == batch_idx_C[i])
            {
                //printf("C->A: %d(%c) %d(%c)\n", i, batch_idx_C[i], j, dense_idx_A[j]);
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
            if (dense_idx_B[j] == batch_idx_C[i])
            {
                //printf("C->B: %d(%c) %d(%c)\n", i, batch_idx_C[i], j, dense_idx_B[j]);
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
            if (dense_idx_C[j] == batch_idx_C[i])
            {
                //printf("C->C: %d(%c) %d(%c)\n", i, batch_idx_C[i], j, dense_idx_C[j]);
                batch_stride_C_C[i] += stride_C[j];
                len_C.erase(len_C.begin()+j);
                stride_C.erase(stride_C.begin()+j);
                dense_idx_C.erase(dense_idx_C.begin()+j);
                ndim_C--;
            }
            else j++;
        }
    }

    varray_view<const T> tensor_A(len_A, nullptr, stride_A);
    varray_view<const T> tensor_B(len_B, nullptr, stride_B);
          varray_view<T> tensor_C(len_C, nullptr, stride_C);

    auto dense_idx_AB = stl_ext::intersection(dense_idx_A, dense_idx_B);
    auto dense_idx_AC = stl_ext::intersection(dense_idx_A, dense_idx_C);
    auto dense_idx_BC = stl_ext::intersection(dense_idx_B, dense_idx_C);
    auto dense_len_AB = stl_ext::select_from(len_A, dense_idx_A, dense_idx_AB);
    auto dense_len_AC = stl_ext::select_from(len_A, dense_idx_A, dense_idx_AC);
    auto dense_len_BC = stl_ext::select_from(len_B, dense_idx_B, dense_idx_BC);

    len_type dense_M = stl_ext::prod(dense_len_AC);
    len_type dense_N = stl_ext::prod(dense_len_BC);
    len_type dense_K = stl_ext::prod(dense_len_AB);

    auto local_flops = 2*dense_K*dense_M*dense_N;

    //printf("(%ld,%ld,%ld)\n", dense_M, dense_N, dense_K);

    #pragma omp parallel for if(outer_threading), schedule(static,1), \
                             firstprivate(tensor_A, tensor_B, tensor_C)
    for (unsigned batch_C = 0;batch_C < C.num_indices();batch_C++)
    {
        T local_beta = beta;

        for (unsigned batch_A = 0;batch_A < A.num_indices();batch_A++)
        {
            bool ok = true;
            for (unsigned i = 0;ok && i < batch_ndim_C;i++)
            {
                for (unsigned j = 0;ok && j < batch_ndim_A;j++)
                {
                    if (batch_idx_C[i] == batch_idx_A[j] &&
                        C.indices()[batch_C][i] !=
                        A.indices()[batch_A][j]) ok = false;
                }
            }
            if (!ok) continue;

            for (unsigned batch_B = 0;batch_B < B.num_indices();batch_B++)
            {
                bool ok = true;
                for (unsigned i = 0;ok && i < batch_ndim_C;i++)
                {
                    for (unsigned j = 0;ok && j < batch_ndim_B;j++)
                    {
                        if (batch_idx_C[i] == batch_idx_B[j] &&
                            C.indices()[batch_C][i] !=
                            B.indices()[batch_B][j]) ok = false;
                    }
                }
                for (unsigned i = 0;ok && i < batch_ndim_A;i++)
                {
                    for (unsigned j = 0;ok && j < batch_ndim_B;j++)
                    {
                        if (batch_idx_A[i] == batch_idx_B[j] &&
                            A.indices()[batch_A][i] !=
                            B.indices()[batch_B][j]) ok = false;
                    }
                }
                if (!ok) continue;

                //printf("idx: %ld %ld %ld\n",
                //       C.indices()[batch_C][0],
                //       C.indices()[batch_C][1],
                //       C.indices()[batch_C][2]);

                const T* local_A = A.data(batch_A);
                const T* local_B = B.data(batch_B);
                      T* local_C = C.data(batch_C);

                for (unsigned i = 0;i < batch_ndim_A;i++)
                {
                    len_type k = A.indices()[batch_A][i];
                    local_A += batch_stride_A_A[i]*k;
                    local_B += batch_stride_A_B[i]*k;
                    local_C += batch_stride_A_C[i]*k;
                }
                for (unsigned i = 0;i < batch_ndim_B;i++)
                {
                    len_type k = B.indices()[batch_B][i];
                    local_A += batch_stride_B_A[i]*k;
                    local_B += batch_stride_B_B[i]*k;
                    local_C += batch_stride_B_C[i]*k;
                }
                for (unsigned i = 0;i < batch_ndim_C;i++)
                {
                    len_type k = C.indices()[batch_C][i];
                    local_A += batch_stride_C_A[i]*k;
                    local_B += batch_stride_C_B[i]*k;
                    local_C += batch_stride_C_C[i]*k;
                }

                //printf("off: %ld %ld %ld\n", local_A-A.data(0),
                //                        local_B-B.data(0),
                //                        local_C-C.data(0));

                tensor_A.data(local_A);
                tensor_B.data(local_B);
                tensor_C.data(local_C);

                flops += local_flops;

                //std::cout << dense_idx_A << " " <<
                //             tensor_A.lengths() << " " <<
                //             tensor_A.dense_strides() << " " <<
                //             dense_idx_B << " " <<
                //             tensor_B.lengths() << " " <<
                //             tensor_B.dense_strides() << " " <<
                //             dense_idx_C << " " <<
                //             tensor_C.lengths() << " " <<
                //             tensor_C.dense_strides() << " " <<
                //             batch_A << " " <<
                //             batch_B << " " <<
                //             batch_C << std::endl;

                if (outer_threading)
                {
                    mult(single, alpha, tensor_A, dense_idx_A.data(),
                                        tensor_B, dense_idx_B.data(),
                            local_beta, tensor_C, dense_idx_C.data());
                }
                else
                {
                    mult(     alpha, tensor_A, dense_idx_A.data(),
                                     tensor_B, dense_idx_B.data(),
                         local_beta, tensor_C, dense_idx_C.data());
                }

                local_beta = T(1);
            }
        }
    }
}

template <typename T>
void contract_batch(T alpha, indexed_varray_view<const T> A, const label_type* idx_A_,
                             indexed_varray_view<const T> B, const label_type* idx_B_,
                    T  beta,       indexed_varray_view<T> C, const label_type* idx_C_)
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

    unsigned batch_ndim_A = A.indexed_dimension();
    unsigned batch_ndim_B = B.indexed_dimension();
    unsigned batch_ndim_C = C.indexed_dimension();

    label_vector idx_A(idx_A_, idx_A_+ndim_A);
    label_vector idx_B(idx_B_, idx_B_+ndim_B);
    label_vector idx_C(idx_C_, idx_C_+ndim_C);

    label_vector dense_idx_A(idx_A_, idx_A_+dense_ndim_A);
    label_vector dense_idx_B(idx_B_, idx_B_+dense_ndim_B);
    label_vector dense_idx_C(idx_C_, idx_C_+dense_ndim_C);

    label_vector batch_idx_A(idx_A_+dense_ndim_A, idx_A_+ndim_A);
    label_vector batch_idx_B(idx_B_+dense_ndim_B, idx_B_+ndim_B);
    label_vector batch_idx_C(idx_C_+dense_ndim_C, idx_C_+ndim_C);

    const auto& len_A = A.lengths();
    const auto& len_B = B.lengths();
    const auto& len_C = C.lengths();

    const auto& dense_len_A = A.dense_lengths();
    const auto& dense_len_B = B.dense_lengths();
    const auto& dense_len_C = C.dense_lengths();

    const auto& batch_len_A = A.indexed_lengths();
    const auto& batch_len_B = B.indexed_lengths();
    const auto& batch_len_C = C.indexed_lengths();

    len_vector stride_A(ndim_A);
    len_vector stride_B(ndim_B);
    len_vector stride_C(ndim_C);
    std::copy_n(A.dense_strides().begin(), dense_ndim_A, stride_A.begin());
    std::copy_n(B.dense_strides().begin(), dense_ndim_B, stride_B.begin());
    std::copy_n(C.dense_strides().begin(), dense_ndim_C, stride_C.begin());

    const auto& dense_stride_A = A.dense_strides();
    const auto& dense_stride_B = B.dense_strides();
    const auto& dense_stride_C = C.dense_strides();

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

    auto batch_len_AB = select_from(len_A, idx_A, batch_idx_AB);
    auto batch_len_AC = select_from(len_A, idx_A, batch_idx_AC);
    auto batch_len_BC = select_from(len_B, idx_B, batch_idx_BC);

    stride_vector off_stride_AB(batch_idx_AB.size()+batch_idx_AC.size()+batch_idx_BC.size());
    stride_vector off_stride_AC(batch_idx_AC.size()+batch_idx_AB.size()+batch_idx_BC.size());
    stride_vector off_stride_BC(batch_idx_BC.size()+batch_idx_AB.size()+batch_idx_AC.size());

    if (!batch_idx_AB.empty())
    stl_ext::prefix_sum(batch_len_AB.begin(), batch_len_AB.end(),
                        off_stride_AB.begin(), 1, std::multiplies<stride_type>());
    if (!batch_idx_AC.empty())
    stl_ext::prefix_sum(batch_len_AC.begin(), batch_len_AC.end(),
                        off_stride_AC.begin(), 1, std::multiplies<stride_type>());
    if (!batch_idx_BC.empty())
    stl_ext::prefix_sum(batch_len_BC.begin(), batch_len_BC.end(),
                        off_stride_BC.begin(), 1, std::multiplies<stride_type>());

    //std::cout << batch_idx_BC << " " << batch_len_BC << " -> " << off_stride_BC << std::endl;

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

    //std::cout << mixed_len_B_BC << " " << mixed_len_C_BC << std::endl;
    //std::cout << mixed_stride_B_BC << " " << mixed_stride_C_BC << std::endl;
    //std::cout << mixed_off_stride_B_BC << " " << mixed_off_stride_C_BC << std::endl;
    //std::cout << batch_off_stride_B_BC << " " << batch_off_stride_C_BC << std::endl;

    matrix<const T*> batch_A({batch_M, batch_K});
    matrix<const T*> batch_B({batch_K, batch_N});
    matrix<      T*> batch_C({batch_M, batch_N});

    viterator<2> it_A_AB(mixed_len_A_AB, mixed_stride_A_AB, mixed_off_stride_A_AB);
    viterator<2> it_A_AC(mixed_len_A_AC, mixed_stride_A_AC, mixed_off_stride_A_AC);

    for (len_type b = 0;b < A.num_indices();b++)
    {
        len_type off_M = 0, off_K = 0;
        for (unsigned i = 0;i < batch_ndim_A;i++)
        {
            off_M += A.indices()[b][i]*batch_off_stride_A_AC[i];
            off_K += A.indices()[b][i]*batch_off_stride_A_AB[i];
        }

        const T* ptr_A = A.data(b);

        while (it_A_AB.next(ptr_A, off_K))
        {
            while (it_A_AC.next(ptr_A, off_M))
            {
                batch_A[off_M][off_K] = ptr_A;
            }
        }
    }

    viterator<2> it_B_AB(mixed_len_B_AB, mixed_stride_B_AB, mixed_off_stride_B_AB);
    viterator<2> it_B_BC(mixed_len_B_BC, mixed_stride_B_BC, mixed_off_stride_B_BC);

    for (len_type b = 0;b < B.num_indices();b++)
    {
        len_type off_K = 0, off_N = 0;
        for (unsigned i = 0;i < batch_ndim_B;i++)
        {
            off_K += B.indices()[b][i]*batch_off_stride_B_AB[i];
            off_N += B.indices()[b][i]*batch_off_stride_B_BC[i];
        }

        const T* ptr_B = B.data(b);

        while (it_B_AB.next(ptr_B, off_K))
        {
            while (it_B_BC.next(ptr_B, off_N))
            {
                batch_B[off_K][off_N] = ptr_B;
            }
        }
    }

    viterator<2> it_C_AC(mixed_len_C_AC, mixed_stride_C_AC, mixed_off_stride_C_AC);
    viterator<2> it_C_BC(mixed_len_C_BC, mixed_stride_C_BC, mixed_off_stride_C_BC);

    for (len_type b = 0;b < C.num_indices();b++)
    {
        len_type off_M = 0, off_N = 0;
        for (unsigned i = 0;i < batch_ndim_C;i++)
        {
            off_M += C.indices()[b][i]*batch_off_stride_C_AC[i];
            off_N += C.indices()[b][i]*batch_off_stride_C_BC[i];
        }

        //printf("C: %ld %ld %ld - %ld %ld\n",
        //       C.indices()[b][0],
        //       C.indices()[b][1],
        //       C.indices()[b][2], off_M, off_N);

        T* ptr_C = C.data(b);

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

    auto batch_max = std::max({batch_M, batch_N, batch_K});
    len_vector nonzero(batch_max+1);

    len_type dense_M = stl_ext::prod(dense_len_AC);
    len_type dense_N = stl_ext::prod(dense_len_BC);
    len_type dense_K = stl_ext::prod(dense_len_AB);

    auto local_flops = 2*dense_K*dense_M*dense_N;

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
                            if (batch_A[m][k] && batch_B[k][n] && batch_C[m][n])
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
                            if (batch_A[m][k] && batch_B[k][n] && batch_C[m][n])
                                nonzero[n]++;
                        }
                    }
                }

                comm.barrier();

                len_type total = stl_ext::sum(nonzero);

                int nt_outer, nt_inner;
                std::tie(nt_outer, nt_inner) =
                    partition_2x2(comm.num_threads(), inout_ratio*total, dense_M*dense_N);

                //if (comm.master()) printf("%ld:%ld -> %d:%d\n", inout_ratio*total, dense_M*dense_N, nt_outer, nt_inner);

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

template <typename T>
void contract_batch2(T alpha, indexed_varray_view<const T> A, const label_type* idx_A_,
                              indexed_varray_view<const T> B, const label_type* idx_B_,
                     T  beta,       indexed_varray_view<T> C, const label_type* idx_C_)
{
    using stl_ext::intersection;
    using stl_ext::exclusion;
    using stl_ext::select_from;
    using stl_ext::permute;

    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();
    unsigned ndim_C = C.dimension();

    unsigned dense_ndim_A = A.dense_dimension();
    unsigned dense_ndim_B = B.dense_dimension();
    unsigned dense_ndim_C = C.dense_dimension();

    unsigned batch_ndim_A = A.indexed_dimension();
    unsigned batch_ndim_B = B.indexed_dimension();
    unsigned batch_ndim_C = C.indexed_dimension();

    label_vector idx_A(idx_A_, idx_A_+ndim_A);
    label_vector idx_B(idx_B_, idx_B_+ndim_B);
    label_vector idx_C(idx_C_, idx_C_+ndim_C);

    label_vector dense_idx_A(idx_A_, idx_A_+dense_ndim_A);
    label_vector dense_idx_B(idx_B_, idx_B_+dense_ndim_B);
    label_vector dense_idx_C(idx_C_, idx_C_+dense_ndim_C);

    label_vector batch_idx_A(idx_A_+dense_ndim_A, idx_A_+ndim_A);
    label_vector batch_idx_B(idx_B_+dense_ndim_B, idx_B_+ndim_B);
    label_vector batch_idx_C(idx_C_+dense_ndim_C, idx_C_+ndim_C);

    const auto& len_A = A.lengths();
    const auto& len_B = B.lengths();
    const auto& len_C = C.lengths();

    const auto& dense_len_A = A.dense_lengths();
    const auto& dense_len_B = B.dense_lengths();
    const auto& dense_len_C = C.dense_lengths();

    const auto& batch_len_A = A.indexed_lengths();
    const auto& batch_len_B = B.indexed_lengths();
    const auto& batch_len_C = C.indexed_lengths();

    len_vector stride_A(ndim_A);
    len_vector stride_B(ndim_B);
    len_vector stride_C(ndim_C);
    std::copy_n(A.dense_strides().begin(), dense_ndim_A, stride_A.begin());
    std::copy_n(B.dense_strides().begin(), dense_ndim_B, stride_B.begin());
    std::copy_n(C.dense_strides().begin(), dense_ndim_C, stride_C.begin());

    const auto& dense_stride_A = A.dense_strides();
    const auto& dense_stride_B = B.dense_strides();
    const auto& dense_stride_C = C.dense_strides();

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

    auto batch_len_AB = select_from(len_A, idx_A, batch_idx_AB);
    auto batch_len_AC = select_from(len_A, idx_A, batch_idx_AC);
    auto batch_len_BC = select_from(len_B, idx_B, batch_idx_BC);

    stride_vector off_stride_AB(batch_idx_AB.size()+batch_idx_AC.size()+batch_idx_BC.size());
    stride_vector off_stride_AC(batch_idx_AC.size()+batch_idx_AB.size()+batch_idx_BC.size());
    stride_vector off_stride_BC(batch_idx_BC.size()+batch_idx_AB.size()+batch_idx_AC.size());

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

    matrix<const T*> batch_A_({batch_M, batch_K});
    matrix<const T*> batch_B_({batch_K, batch_N});
    matrix<      T*> batch_C_({batch_M, batch_N});

    matrix_view<const T*> batch_A(batch_A_);
    matrix_view<const T*> batch_B(batch_B_);
    matrix_view<      T*> batch_C(batch_C_);

    viterator<2> it_A_AB(mixed_len_A_AB, mixed_stride_A_AB, mixed_off_stride_A_AB);
    viterator<2> it_A_AC(mixed_len_A_AC, mixed_stride_A_AC, mixed_off_stride_A_AC);

    for (len_type b = 0;b < A.num_indices();b++)
    {
        len_type off_M = 0, off_K = 0;
        for (unsigned i = 0;i < batch_ndim_A;i++)
        {
            off_M += A.indices()[b][i]*batch_off_stride_A_AC[i];
            off_K += A.indices()[b][i]*batch_off_stride_A_AB[i];
        }

        //printf("A: %ld %ld %ld - %ld %ld\n",
        //       A.indices()[b][0],
        //       A.indices()[b][1],
        //       A.indices()[b][2], off_M, off_K);

        const T* ptr_A = A.data(b);

        while (it_A_AB.next(ptr_A, off_K))
        {
            while (it_A_AC.next(ptr_A, off_M))
            {
                batch_A[off_M][off_K] = ptr_A;
            }
        }
    }

    viterator<2> it_B_AB(mixed_len_B_AB, mixed_stride_B_AB, mixed_off_stride_B_AB);
    viterator<2> it_B_BC(mixed_len_B_BC, mixed_stride_B_BC, mixed_off_stride_B_BC);

    for (len_type b = 0;b < B.num_indices();b++)
    {
        len_type off_K = 0, off_N = 0;
        for (unsigned i = 0;i < batch_ndim_B;i++)
        {
            off_K += B.indices()[b][i]*batch_off_stride_B_AB[i];
            off_N += B.indices()[b][i]*batch_off_stride_B_BC[i];
        }

        //printf("B: %ld %ld %ld - %ld %ld\n",
        //       B.indices()[b][0],
        //       B.indices()[b][1],
        //       B.indices()[b][2], off_K, off_N);

        const T* ptr_B = B.data(b);

        while (it_B_AB.next(ptr_B, off_K))
        {
            while (it_B_BC.next(ptr_B, off_N))
            {
                batch_B[off_K][off_N] = ptr_B;
            }
        }
    }

    viterator<2> it_C_AC(mixed_len_C_AC, mixed_stride_C_AC, mixed_off_stride_C_AC);
    viterator<2> it_C_BC(mixed_len_C_BC, mixed_stride_C_BC, mixed_off_stride_C_BC);

    for (len_type b = 0;b < C.num_indices();b++)
    {
        len_type off_M = 0, off_N = 0;
        for (unsigned i = 0;i < batch_ndim_C;i++)
        {
            off_M += C.indices()[b][i]*batch_off_stride_C_AC[i];
            off_N += C.indices()[b][i]*batch_off_stride_C_BC[i];
        }

        //printf("C: %ld %ld %ld %ld - %ld %ld\n",
        //       C.indices()[b][0],
        //       C.indices()[b][1],
        //       C.indices()[b][2],
        //       C.indices()[b][3], off_M, off_N);

        T* ptr_C = C.data(b);

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

    auto reorder_AC = detail::sort_by_stride(dense_stride_C_AC, dense_stride_A_AC);
    auto reorder_BC = detail::sort_by_stride(dense_stride_C_BC, dense_stride_B_BC);
    auto reorder_AB = detail::sort_by_stride(dense_stride_A_AB, dense_stride_B_AB);

    permute(dense_len_AB, reorder_AB);
    permute(dense_len_AC, reorder_AC);
    permute(dense_len_BC, reorder_BC);
    permute(dense_stride_A_AB, reorder_AB);
    permute(dense_stride_B_AB, reorder_AB);
    permute(dense_stride_A_AC, reorder_AC);
    permute(dense_stride_C_AC, reorder_AC);
    permute(dense_stride_B_BC, reorder_BC);
    permute(dense_stride_C_BC, reorder_BC);

    len_type dense_M = stl_ext::prod(dense_len_AC);
    len_type dense_N = stl_ext::prod(dense_len_BC);
    len_type dense_K = stl_ext::prod(dense_len_AB);

    auto local_flops = 2*dense_K*dense_M*dense_N;

    std::vector<slot<int, -1>> slot(batch_M*batch_N);

    const config& cfg = get_default_config();
    const bool row_major = cfg.gemm_row_major.value<T>();
    const bool transpose = (row_major ? !dense_stride_C_AC.empty() && dense_stride_C_AC[0] == 1
                                      : !dense_stride_C_BC.empty() && dense_stride_C_BC[0] == 1);

    if (transpose)
    {
        using std::swap;
        swap(dense_M, dense_N);
        swap(batch_M, batch_N);
        swap(dense_len_AC, dense_len_BC);
        swap(dense_stride_A_AC, dense_stride_B_BC);
        swap(dense_stride_A_AB, dense_stride_B_AB);
        swap(dense_stride_C_AC, dense_stride_C_BC);
        swap(batch_A, batch_B);
        batch_A.transpose();
        batch_B.transpose();
        batch_C.transpose();
    }

    std::atomic<len_type> nonzero = {0}, quasi = {0};

    parallelize
    (
        [&](const communicator& comm)
        {
            if (batch_K > 1 && dense_K <= dense_M && dense_K <= dense_N)
            {
                //if (comm.master()) printf("K algorithm\n");

                scatter_tensor_matrix<T> at(dense_len_AC, 1,
                                            dense_len_AB, 1,
                                            nullptr,
                                            dense_stride_A_AC, nullptr,
                                            dense_stride_A_AB, nullptr);

                scatter_tensor_matrix<T> bt(dense_len_AB, 1,
                                            dense_len_BC, 1,
                                            nullptr,
                                            dense_stride_B_AB, nullptr,
                                            dense_stride_B_BC, nullptr);

                tensor_matrix<T> ct(dense_len_AC,
                                    dense_len_BC,
                                    nullptr,
                                    dense_stride_C_AC,
                                    dense_stride_C_BC);

                len_type nonzero_local = 0;

                len_type mn_min, mn_max;
                std::tie(mn_min, mn_max, std::ignore) =
                    comm.distribute_over_threads(batch_M*batch_N);

                for (len_type mn = mn_min;mn < mn_max;mn++)
                {
                    if (batch_C.data()[mn]) nonzero_local++;
                }

                nonzero += nonzero_local;

                comm.barrier();

                int nt_outer, nt_inner;
                std::tie(nt_outer, nt_inner) =
                    partition_2x2(comm.num_threads(), inout_ratio*nonzero, dense_M*dense_N);

                //if (comm.master()) printf("%ld:%ld -> %d:%d\n", inout_ratio*nonzero, dense_M*dense_N, nt_outer, nt_inner);

                communicator subcomm = comm.gang(TCI_EVENLY, nt_outer);
                unsigned gid = subcomm.gang_num();

                auto tc = make_gemm_thread_config<T>(cfg, subcomm.num_threads(),
                                                     dense_M, dense_N, dense_K);

                stride_vector scatter_A(batch_K);
                stride_vector scatter_B(batch_K);

                const T* A0 = nullptr;
                const T* B0 = nullptr;

                for (len_type m = 0, mn = 0;m < batch_M;m++)
                {
                    for (len_type n = 0;n < batch_N;n++, mn++)
                    {
                        auto C = batch_C[m][n];
                        if (!C || !slot[mn].try_fill(gid)) continue;

                        len_type knz = 0;
                        for (len_type k = 0;k < batch_K;k++)
                        {
                            auto A = batch_A[m][k];
                            auto B = batch_B[k][n];
                            if (!A || !B) continue;

                            if (knz == 0)
                            {
                                A0 = A;
                                B0 = B;
                            }
                            else
                            {
                                scatter_A[knz] = A-A0;
                                scatter_B[knz] = B-B0;
                            }
                            knz++;
                        }

                        if (knz == 0) continue;

                        if (subcomm.master()) flops += local_flops*knz;

                        ct.data(C);

                        at.scatter_length(1, knz);
                        bt.scatter_length(0, knz);
                        at.scatter(1, scatter_A.data());
                        bt.scatter(0, scatter_B.data());
                        at.data(const_cast<T*>(A0));
                        bt.data(const_cast<T*>(B0));

                        internal::TensorGEMM gemm;

                        step<0>(gemm).distribute = tc.jc_nt;
                        step<4>(gemm).distribute = tc.ic_nt;
                        step<8>(gemm).distribute = tc.jr_nt;
                        step<9>(gemm).distribute = tc.ir_nt;

                        gemm(subcomm, cfg, alpha, at, bt, beta, ct);
                    }
                }
            }
            else if (batch_N > 1 && dense_N <= dense_M)
            {
                //if (comm.master()) printf("N algorithm\n");

                tensor_matrix<T> at(dense_len_AC,
                                    dense_len_AB,
                                    nullptr,
                                    dense_stride_A_AC,
                                    dense_stride_A_AB);

                scatter_tensor_matrix<T> bt(dense_len_AB, 1,
                                            dense_len_BC, 1,
                                            nullptr,
                                            dense_stride_B_AB, nullptr,
                                            dense_stride_B_BC, nullptr);

                scatter_tensor_matrix<T> ct(dense_len_AC, 1,
                                            dense_len_BC, 1,
                                            nullptr,
                                            dense_stride_C_AC, nullptr,
                                            dense_stride_C_BC, nullptr);

                len_type nonzero_local = 0;
                len_type quasi_local = 0;

                len_type m_min, m_max;
                std::tie(m_min, m_max, std::ignore) =
                    comm.distribute_over_threads(batch_M);

                for (len_type m = m_min;m < m_max;m++)
                {
                    bool found = false;
                    for (len_type n = 0;n < batch_N;n++)
                    {
                        if (batch_C[m][n])
                        {
                            quasi_local++;
                            found = true;
                        }
                    }
                    if (found) nonzero_local++;
                }

                nonzero += nonzero_local;
                quasi += quasi_local;

                comm.barrier();

                len_type quasi_N = (quasi+1)/nonzero;

                int nt_outer, nt_inner;
                std::tie(nt_outer, nt_inner) =
                    partition_2x2(comm.num_threads(), inout_ratio*nonzero, dense_M*dense_N*quasi_N);

                //if (comm.master()) printf("%d %ld:%ld -> %d:%d %d\n", comm.thread_num(), inout_ratio*nonzero, dense_M*dense_N*quasi_N, nt_outer, nt_inner, quasi_N);

                communicator subcomm = comm.gang(TCI_EVENLY, nt_outer);
                unsigned gid = subcomm.gang_num();

                stride_vector scatter_B(batch_N);
                stride_vector scatter_C(batch_N);

                const T* B0 = nullptr;
                const T* C0 = nullptr;

                for (len_type k = 0;k < batch_K;k++)
                {
                    for (len_type m = 0;m < batch_M;m++)
                    {
                        auto A = batch_A[m][k];
                        if (!A || !slot[m].try_fill(gid)) continue;

                        len_type nnz = 0;
                        for (len_type n = 0;n < batch_N;n++)
                        {
                            auto B = batch_B[k][n];
                            auto C = batch_C[m][n];
                            if (!B || !C) continue;

                            if (nnz == 0)
                            {
                                B0 = B;
                                C0 = C;
                            }
                            else
                            {
                                scatter_B[nnz] = B-B0;
                                scatter_C[nnz] = C-C0;
                            }
                            nnz++;
                        }

                        if (nnz == 0) continue;

                        if (subcomm.master()) flops += local_flops*nnz;

                        at.data(const_cast<T*>(A));

                        bt.scatter_length(1, nnz);
                        ct.scatter_length(1, nnz);
                        bt.scatter(1, scatter_B.data());
                        ct.scatter(1, scatter_C.data());
                        bt.data(const_cast<T*>(B0));
                        ct.data(const_cast<T*>(C0));

                        internal::TensorGEMM gemm;

                        auto tc = make_gemm_thread_config<T>(cfg, subcomm.num_threads(),
                                                             dense_M, dense_N*nnz, dense_K);

                        step<0>(gemm).distribute = tc.jc_nt;
                        step<4>(gemm).distribute = tc.ic_nt;
                        step<8>(gemm).distribute = tc.jr_nt;
                        step<9>(gemm).distribute = tc.ir_nt;

                        gemm(subcomm, cfg, alpha, at, bt, beta, ct);
                    }

                    comm.barrier();
                }
            }
            else
            {
                //if (comm.master()) printf("M algorithm\n");

                scatter_tensor_matrix<T> at(dense_len_AC, 1,
                                            dense_len_AB, 1,
                                            nullptr,
                                            dense_stride_A_AC, nullptr,
                                            dense_stride_A_AB, nullptr);

                tensor_matrix<T> bt(dense_len_AB,
                                    dense_len_BC,
                                    nullptr,
                                    dense_stride_B_AB,
                                    dense_stride_B_BC);

                scatter_tensor_matrix<T> ct(dense_len_AC, 1,
                                            dense_len_BC, 1,
                                            nullptr,
                                            dense_stride_C_AC, nullptr,
                                            dense_stride_C_BC, nullptr);

                len_type nonzero_local = 0;
                len_type quasi_local = 0;

                len_type n_min, n_max;
                std::tie(n_min, n_max, std::ignore) =
                    comm.distribute_over_threads(batch_N);

                for (len_type n = n_min;n < n_max;n++)
                {
                    bool found = false;
                    for (len_type m = 0;m < batch_M;m++)
                    {
                        if (batch_C[m][n])
                        {
                            quasi_local++;
                            found = true;
                        }
                    }
                    if (found) nonzero_local++;
                }

                nonzero += nonzero_local;
                quasi += quasi_local;

                comm.barrier();

                len_type quasi_M = (quasi+1)/nonzero;

                int nt_outer, nt_inner;
                std::tie(nt_outer, nt_inner) =
                    partition_2x2(comm.num_threads(), inout_ratio*nonzero, dense_M*dense_N*quasi_M);

                //if (comm.master()) printf("%ld:%ld -> %d:%d\n", inout_ratio*nonzero, dense_M*dense_N*quasi_M, nt_outer, nt_inner);

                communicator subcomm = comm.gang(TCI_EVENLY, nt_outer);
                unsigned gid = subcomm.gang_num();

                stride_vector scatter_A(batch_M);
                stride_vector scatter_C(batch_M);

                const T* A0 = nullptr;
                const T* C0 = nullptr;

                for (len_type k = 0;k < batch_K;k++)
                {
                    for (len_type n = 0;n < batch_N;n++)
                    {
                        auto B = batch_B[k][n];
                        if (!B || !slot[n].try_fill(gid)) continue;

                        len_type mnz = 0;
                        for (len_type m = 0;m < batch_M;m++)
                        {
                            auto A = batch_A[m][k];
                            auto C = batch_C[m][n];
                            if (!A || !C) continue;

                            if (mnz == 0)
                            {
                                A0 = A;
                                C0 = C;
                            }
                            else
                            {
                                scatter_A[mnz] = A-A0;
                                scatter_C[mnz] = C-C0;
                            }
                            mnz++;
                        }

                        if (mnz == 0) continue;

                        if (subcomm.master()) flops += local_flops*mnz;

                        bt.data(const_cast<T*>(B));

                        at.scatter_length(0, mnz);
                        ct.scatter_length(0, mnz);
                        at.scatter(0, scatter_A.data());
                        ct.scatter(0, scatter_C.data());
                        at.data(const_cast<T*>(A0));
                        ct.data(const_cast<T*>(C0));

                        internal::TensorGEMM gemm;

                        auto tc = make_gemm_thread_config<T>(cfg, subcomm.num_threads(),
                                                             dense_M*mnz, dense_N, dense_K);

                        step<0>(gemm).distribute = tc.jc_nt;
                        step<4>(gemm).distribute = tc.ic_nt;
                        step<8>(gemm).distribute = tc.jr_nt;
                        step<9>(gemm).distribute = tc.ir_nt;

                        gemm(subcomm, cfg, alpha, at, bt, beta, ct);
                    }

                    subcomm.barrier();
                }
            }
        },
        tblis_get_num_threads()
    );
}

}

#endif
