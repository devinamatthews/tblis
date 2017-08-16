#include "dot.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dot(const communicator& comm, const config& cfg,
         const len_vector& len_AB,
         bool conj_A, const T* A, const stride_vector& stride_A_AB,
         bool conj_B, const T* B, const stride_vector& stride_B_AB,
         T& result)
{
    (void)cfg;

    viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);

    len_type n = stl_ext::prod(len_AB);

    len_type n_min, n_max;
    std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(n);

    T local_result = T();

    if (conj_A) conj_B = !conj_B;

    iter_AB.position(n_min, A, B);

    for (len_type i = n_min;i < n_max;i++)
    {
        iter_AB.next(A, B);
        local_result += (*A)*(conj_B ? conj(*B) : *B);
    }

    len_type dummy = 0;
    reduce(comm, REDUCE_SUM, local_result, dummy);
    if (comm.master())
        result = (conj_A ? conj(local_result) : local_result);

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void dot(const communicator& comm, const config& cfg, \
                  const len_vector& len_AB, \
                  bool conj_A, const T* A, const stride_vector& stride_A_AB, \
                  bool conj_B, const T* B, const stride_vector& stride_B_AB, \
                  T& result);
#include "configs/foreach_type.h"

}
}
