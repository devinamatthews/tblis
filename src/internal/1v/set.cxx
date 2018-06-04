#include "set.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void set(const communicator& comm, const config& cfg, len_type n,
         T alpha, T* A, stride_type inc_A)
{
    comm.distribute_over_threads(n,
    [&](len_type n_min, len_type n_max)
    {
        cfg.shift_ukr.call<T>(n_max-n_min, alpha, T(0), false, A+n_min*inc_A, inc_A);
    });
}

#define FOREACH_TYPE(T) \
template void set(const communicator& comm, const config& cfg, len_type n, \
                  T alpha, T* A, stride_type inc_A);
#include "configs/foreach_type.h"

}
}
