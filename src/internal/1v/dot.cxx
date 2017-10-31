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
    atomic_accumulator<T> local_result;

    comm.distribute_over_threads(n,
    [&](len_type n_min, len_type n_max)
    {
        T micro_result = T();

        cfg.dot_ukr.call<T>(n_max-n_min,
                            conj_A, A + n_min*inc_A, inc_A,
                            conj_B, B + n_min*inc_B, inc_B, micro_result);

        local_result += micro_result;
    });

    reduce(comm, local_result);
    if (comm.master()) result = local_result;

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void dot(const communicator& comm, const config& cfg, len_type n, \
                  bool conj_A, const T* A, stride_type inc_A, \
                  bool conj_B, const T* B, stride_type inc_B, T& result);
#include "configs/foreach_type.h"

}
}
