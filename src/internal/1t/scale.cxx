#include "scale.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void scale(const communicator& comm, const config& cfg,
           const std::vector<len_type>& len_A,
           T alpha, bool conj_A, T* A, const std::vector<stride_type>& stride_A)
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

    iter_A.position(n_min, A);
    A += m_min*stride0;

    for (len_type i = n_min;i < n_max;i++)
    {
        iter_A.next(A);
        cfg.scale_ukr.call<T>(m_max-m_min,
                              alpha, conj_A, A, stride0);
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void scale(const communicator& comm, const config& cfg, \
                    const std::vector<len_type>& len_A, \
                    T alpha, bool conj_A, T* A, const std::vector<stride_type>& stride_A);
#include "configs/foreach_type.h"

}
}
