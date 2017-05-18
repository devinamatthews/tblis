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
    bool empty = len_A.size() == 0;

    len_type len0 = (empty ? 1 : len_A[0]);
    std::vector<len_type> len1(len_A.begin() + !empty, len_A.end());

    stride_type stride0 = (empty ? 1 : stride_A[0]);
    std::vector<len_type> stride1(stride_A.begin() + !empty, stride_A.end());

    MArray::viterator<1> iter_A(len1, stride1);
    len_type n = stl_ext::prod(len1);

    len_type m_min, m_max, n_min, n_max;
    std::tie(m_min, m_max, std::ignore,
             n_min, n_max, std::ignore) =
        comm.distribute_over_threads_2d(len0, n);

    T local_result;
    len_type local_idx;
    reduce_init(op, local_result, local_idx);

    auto A0 = A;
    iter_A.position(n_min, A);
    A += m_min*stride0;

    for (len_type i = n_min;i < n_max;i++)
    {
        auto old_idx = local_idx;
        local_idx = -1;

        iter_A.next(A);
        cfg.reduce_ukr.call<T>(op, m_max-m_min, A, stride0, local_result, local_idx);

        if (local_idx != -1) local_idx += A-A0;
        else local_idx = old_idx;
    }

    reduce(comm, op, local_result, local_idx);

    if (comm.master())
    {
        result = local_result;
        idx = local_idx;
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void reduce(const communicator& comm, const config& cfg, reduce_t op, \
                     const std::vector<len_type>& len_A, \
                     const T* A, const std::vector<stride_type>& stride_A, \
                     T& result, len_type& idx);
#include "configs/foreach_type.h"

}
}
