#include "dot.h"

#include "util/tensor.hpp"
#include "configs/configs.hpp"

namespace tblis
{

void dot_int(const communicator& comm, const config& cfg,
             const tblis_tensor& A, const label_type* idx_A_,
             const tblis_tensor& B, const label_type* idx_B_,
             tblis_scalar& result)
{
    TBLIS_ASSERT(A.type == B.type);
    TBLIS_ASSERT(A.type == result.type);

    int ndim_A = A.ndim;
    std::vector<len_type> len_A;
    std::vector<stride_type> stride_A;
    std::vector<label_type> idx_A;
    diagonal(ndim_A, A.len, A.stride, idx_A_,
             len_A, stride_A, idx_A);

    int ndim_B = B.ndim;
    std::vector<len_type> len_B;
    std::vector<stride_type> stride_B;
    std::vector<label_type> idx_B;
    diagonal(ndim_B, B.len, B.stride, idx_B_,
             len_B, stride_B, idx_B);

    auto idx_AB = stl_ext::intersection(idx_A, idx_B);
    auto len_AB = stl_ext::select_from(len_A, idx_A, idx_AB);
    TBLIS_ASSERT(len_AB == stl_ext::select_from(len_B, idx_B, idx_AB));
    auto stride_A_AB = stl_ext::select_from(stride_A, idx_A, idx_AB);
    auto stride_B_AB = stl_ext::select_from(stride_B, idx_B, idx_AB);

    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB);
    auto len_A_only = stl_ext::select_from(len_A, idx_A, idx_A_only);
    auto stride_A_only = stl_ext::select_from(stride_A, idx_A, idx_A_only);

    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB);
    auto len_B_only = stl_ext::select_from(len_B, idx_B, idx_B_only);
    auto stride_B_only = stl_ext::select_from(stride_B, idx_B, idx_B_only);

    fold(len_AB, idx_AB, stride_A_AB, stride_B_AB);
    fold(len_A_only, idx_A_only, stride_A_only);
    fold(len_B_only, idx_B_only, stride_B_only);

    int ndim_AB = idx_AB.size();
    ndim_A = idx_A_only.size();
    ndim_B = idx_B_only.size();

    MArray::viterator<1> iter_A(len_A_only, stride_A_only);
    MArray::viterator<1> iter_B(len_B_only, stride_B_only);
    MArray::viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);

    len_type n = stl_ext::prod(len_AB);

    len_type n_min, n_max;
    std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(n);

    TBLIS_WITH_TYPE_AS(A.type, T,
    {
        T dot = T();

        bool conj_A = A.conj;
        bool conj_B = (conj_A ? !B.conj : B.conj);

        const T* TBLIS_RESTRICT A_ = (const T*)A.data;
        const T* TBLIS_RESTRICT B_ = (const T*)B.data;

        iter_AB.position(n_min, A_, B_);

        for (len_type i = n_min;i < n_max;i++)
        {
            iter_AB.next(A_, B_);

            T sum_A = T();
            T sum_B = T();
            while (iter_A.next(A_)) sum_A += *A_;
            while (iter_B.next(B_)) sum_B += *B_;

            if (B.conj)
            {
                dot += sum_A*conj(sum_B);
            }
            else
            {
                dot += sum_A*sum_B;
            }
        }

        result.get<T>() = (conj_A ? conj(dot) : dot);
    })
}

extern "C"
{

void tblis_tensor_dot(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_tensor* A, const label_type* idx_A,
                      const tblis_tensor* B, const label_type* idx_B,
                      tblis_scalar* result)
{
    parallelize_if(dot_int, comm, get_config(cfg),
                   *A, idx_A, *B, idx_B, *result);
}

}

}
