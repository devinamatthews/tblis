#include "reduce.h"

#include "util/tensor.hpp"
#include "configs/configs.hpp"

namespace tblis
{

void reduce_int(const communicator& comm, const config& cfg,
                reduce_t op, const tblis_tensor& A, const label_type* idx_A_,
                tblis_scalar& result, len_type& idx)
{
    TBLIS_ASSERT(A.type == result.type);

    int ndim_A = A.ndim;
    std::vector<len_type> len_A(ndim_A);
    std::vector<stride_type> stride_A(ndim_A);
    std::vector<label_type> idx_A(ndim_A);
    diagonal(ndim_A, A.len, A.stride, idx_A_,
             len_A.data(), stride_A.data(), idx_A.data());

    fold(len_A, idx_A, stride_A);
    ndim_A = idx_A.size();

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
        reduce_init(op, result.get<T>(), idx);

        const T* TBLIS_RESTRICT A_ = (T*)A.data + m_min*stride0;
        iter_A.position(n_min, A_);

        while (iter_A.next(A_))
        {
            cfg.reduce_ukr.call<T>(op, m_max-m_min, A_, stride0,
                                   result.get<T>(), idx);
        }

        reduce(comm, op, result.get<T>(), idx);
    })
}

extern "C"
{

void tblis_tensor_reduce(const tblis_comm* comm, const tblis_config* cfg,
                         reduce_t op, const tblis_tensor* A, const label_type* idx_A,
                         tblis_scalar* result, len_type* idx)
{
    parallelize_if(reduce_int, comm, get_config(cfg),
                   op, *A, idx_A, *result, *idx);
}

}

}
