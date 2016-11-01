#include "add.h"

#include "util/tensor.hpp"
#include "configs/configs.hpp"

namespace tblis
{

void add_int(const communicator& comm, const config& cfg,
             const tblis_tensor& A, const label_type* idx_A_,
                   tblis_tensor& B, const label_type* idx_B_)
{
    TBLIS_ASSERT(A.type == B.type);

    int ndim_A = A.ndim;
    std::vector<len_type> len_A(ndim_A);
    std::vector<stride_type> stride_A(ndim_A);
    std::vector<label_type> idx_A(ndim_A);
    diagonal(ndim_A, A.len, A.stride, idx_A_,
             len_A.data(), stride_A.data(), idx_A.data());

    int ndim_B = B.ndim;
    std::vector<len_type> len_B(ndim_B);
    std::vector<stride_type> stride_B(ndim_B);
    std::vector<label_type> idx_B(ndim_B);
    diagonal(ndim_B, B.len, B.stride, idx_B_,
             len_B.data(), stride_B.data(), idx_B.data());

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

    if (idx_A_only.empty() && idx_B_only.empty())
    {
        stride_type stride_A0 = stride_A_AB[0];
        stride_type stride_B0 = stride_B_AB[0];
        len_type len0 = len_AB[0];

        stl_ext::rotate(stride_A_AB, 1);
        stl_ext::rotate(stride_B_AB, 1);
        stl_ext::rotate(len_AB, 1);

        stride_B_AB.resize(ndim_AB-1);
        stride_A_AB.resize(ndim_AB-1);
        len_AB.resize(ndim_AB-1);

        MArray::viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
        len_type n = stl_ext::prod(len_AB);

        len_type m_min, m_max, n_min, n_max;
        std::tie(m_min, m_max, std::ignore,
                 n_min, n_max, std::ignore) =
            comm.distribute_over_threads_2d(len0, n);

        TBLIS_WITH_TYPE_AS(A.type, T,
        {
            const T* TBLIS_RESTRICT A_ = (const T*)A.data + m_min*stride_A0;
                  T* TBLIS_RESTRICT B_ =       (T*)B.data + m_min*stride_B0;
            iter_AB.position(n_min, A_, B_);

            for (len_type i = n_min;i < n_max;i++)
            {
                iter_AB.next(A_, B_);
                cfg.add_ukr.call<T>(m_max-m_min,
                                    A.alpha<T>(), A.conj, A_, stride_A0,
                                    B.alpha<T>(), B.conj, B_, stride_B0);
            }

            B.alpha<T>() = T(1);
            B.conj = false;
        })
    }
    else
    {
        len_type n = stl_ext::prod(len_AB);

        len_type n_min, n_max;
        std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(n);

        TBLIS_WITH_TYPE_AS(A.type, T,
        {
            const T* TBLIS_RESTRICT A_ = (const T*)A.data;
                  T* TBLIS_RESTRICT B_ =       (T*)B.data;

            iter_AB.position(n_min, A_, B_);

            for (len_type i = n_min;i < n_max;i++)
            {
                iter_AB.next(A_, B_);

                T alpha = A.alpha<T>();
                T  beta = B.alpha<T>();

                T sum_A = T();
                while (iter_A.next(A_)) sum_A += *A_;
                sum_A = A.alpha<T>()*(A.conj ? conj(sum_A) : sum_A);

                TBLIS_SPECIAL_CASE(is_complex<T>::value && B.conj,
                TBLIS_SPECIAL_CASE(beta == T(0),
                {
                    while (iter_B.next(B_))
                    {
                        *B_ = sum_A + beta*(B.conj ? conj(*B_) : *B_);
                    }
                }
                ))
            }

            B.alpha<T>() = T(1);
            B.conj = false;
        })
    }
}

extern "C"
{

void tblis_tensor_add(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_tensor* A, const label_type* idx_A,
                            tblis_tensor* B, const label_type* idx_B)
{
    parallelize_if(add_int, comm, get_config(cfg),
                   *A, idx_A, *B, idx_B);
}

}

}
