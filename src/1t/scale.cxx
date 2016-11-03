#include "scale.h"

#include "util/tensor.hpp"
#include "configs/configs.hpp"

namespace tblis
{

void scale_int(const communicator& comm, const config& cfg,
               tblis_tensor& A, const label_type* idx_A_)
{
    int ndim_A = A.ndim;
    std::vector<len_type> len_A;
    std::vector<stride_type> stride_A;
    std::vector<label_type> idx_A;
    diagonal(ndim_A, A.len, A.stride, idx_A_,
             len_A, stride_A, idx_A);

    fold(len_A, idx_A, stride_A);
    ndim_A = idx_A.size();

    if (ndim_A == 0)
    {
        TBLIS_WITH_TYPE_AS(A.type, T,
        {
            T* TBLIS_RESTRICT A_ = (T*)A.data;
            *A_ = A.alpha<T>()*(A.conj ? conj(*A_) : *A_);
            A.alpha<T>() = T(1);
            A.conj = false;
        })
        return;
    }

    stride_type stride0 = stride_A[0];
    len_type len0 = len_A[0];

    stl_ext::rotate(stride_A, 1);
    stl_ext::rotate(len_A, 1);

    stride_A.resize(ndim_A-1);
    len_A.resize(ndim_A-1);

    MArray::viterator<1> iter_A(len_A, stride_A);
    len_type n = stl_ext::prod(len_A);

    len_type m_min, m_max, n_min, n_max;
    std::tie(m_min, m_max, std::ignore,
             n_min, n_max, std::ignore) =
        comm.distribute_over_threads_2d(len0, n);

    TBLIS_WITH_TYPE_AS(A.type, T,
    {
        T* TBLIS_RESTRICT A_ = (T*)A.data + m_min*stride0;
        iter_A.position(n_min, A_);

        for (len_type i = n_min;i < n_max;i++)
        {
            iter_A.next(A_);
            cfg.scale_ukr.call<T>(m_max-m_min,
                                  A.alpha<T>(), A.conj, A_, stride0);
        }

        A.alpha<T>() = T(1);
        A.conj = false;
    })
}

extern "C"
{

void tblis_tensor_scale(const tblis_comm* comm, const tblis_config* cfg,
                        tblis_tensor* A, const label_type* idx_A)
{
    parallelize_if(scale_int, comm, get_config(cfg),
                   *A, idx_A);
}

}

}
