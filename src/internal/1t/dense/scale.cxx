#include "scale.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void scale(const communicator& comm, const config& cfg,
           const len_vector& len_A,
           T alpha, bool conj_A, T* A, const stride_vector& stride_A)
{
    bool empty = len_A.size() == 0;

    len_type n0 = (empty ? 1 : len_A[0]);
    len_vector len1(len_A.begin() + !empty, len_A.end());
    len_type n1 = stl_ext::prod(len1);

    stride_type stride0 = (empty ? 1 : stride_A[0]);
    len_vector stride1(stride_A.begin() + !empty, stride_A.end());

    comm.distribute_over_threads(tci::range(n0).chunk(1000),
                                 tci::range(n1).chunk(1000/n0),
    [&](len_type n0_min, len_type n0_max, len_type n1_min, len_type n1_max)
    {
        auto A1 = A;

        viterator<1> iter_A(len1, stride1);
        iter_A.position(n1_min, A1);

        A1 += n0_min*stride0;

        for (len_type i = n1_min;i < n1_max;i++)
        {
            iter_A.next(A1);
            cfg.scale_ukr.call<T>(n0_max-n0_min,
                                  alpha, conj_A, A1, stride0);
        }
    });

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void scale(const communicator& comm, const config& cfg, \
                    const len_vector& len_A, \
                    T alpha, bool conj_A, T* A, const stride_vector& stride_A);
#include "configs/foreach_type.h"

}
}
