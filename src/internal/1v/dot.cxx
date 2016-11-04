#include "dot.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dot(const communicator& comm, const config& cfg, len_type n,
         bool conj_A, const T* A, stride_type inc_A,
         bool conj_B, const T* B, stride_type inc_B, T& result)
{
    len_type n_min, n_max;
    std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(n);

    result = T();

    cfg.dot_ukr.call<T>(n_max-n_min,
                        conj_A, A + n_min*inc_A, inc_A,
                        conj_B, B + n_min*inc_B, inc_B, result);

    len_type dummy;
    reduce(comm, REDUCE_SUM, result, dummy);
}

#define FOREACH_TYPE(T) \
template void dot(const communicator& comm, const config& cfg, len_type n, \
                  bool conj_A, const T* A, stride_type inc_A, \
                  bool conj_B, const T* B, stride_type inc_B, T& result);
#include "configs/foreach_type.h"

}
}
