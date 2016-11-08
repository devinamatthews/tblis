#include "reduce.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void reduce(const communicator& comm, const config& cfg, reduce_t op,
            const std::vector<len_type>& len_A,
            const T* A, const std::vector<stride_type>& stride_A,
            T& result, len_type& idx)
{
    if (len_A.size() == 0)
    {
        idx = 0;
        switch (op)
        {
            case REDUCE_SUM:
            case REDUCE_MIN:
            case REDUCE_MAX:
                result = *A;
                break;
            case REDUCE_SUM_ABS:
            case REDUCE_NORM_2:
            case REDUCE_MIN_ABS:
            case REDUCE_MAX_ABS:
                result = std::abs(*A);
                break;
        }
        return;
    }

    len_type len0 = len_A[0];
    std::vector<len_type> len1(len_A.begin()+1, len_A.end());

    stride_type stride0 = stride_A[0];
    std::vector<len_type> stride1(stride_A.begin()+1, stride_A.end());

    MArray::viterator<1> iter_A(len1, stride1);
    len_type n = stl_ext::prod(len1);

    len_type m_min, m_max, n_min, n_max;
    std::tie(m_min, m_max, std::ignore,
             n_min, n_max, std::ignore) =
        comm.distribute_over_threads_2d(len0, n);

    reduce_init(op, result, idx);

    iter_A.position(n_min, A);

    for (len_type i = n_min;i < n_max;i++)
    {
        iter_A.next(A);
        cfg.reduce_ukr.call<T>(op, m_max-m_min, A, stride0, result, idx);
    }

    reduce(comm, op, result, idx);
}

#define FOREACH_TYPE(T) \
template void reduce(const communicator& comm, const config& cfg, reduce_t op, \
                     const std::vector<len_type>& len_A, \
                     const T* A, const std::vector<stride_type>& stride_A, \
                     T& result, len_type& idx);
#include "configs/foreach_type.h"

}
}
