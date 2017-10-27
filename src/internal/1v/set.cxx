#include "set.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void set(const communicator& comm, const config& cfg, len_type n,
         T alpha, T* A, stride_type inc_A)
{
    comm.distribute_over_threads(tci::range(n).chunk(1000),
    [&](len_type n_min, len_type n_max)
    {
        cfg.set_ukr.call<T>(n_max-n_min, alpha, A+n_min*inc_A, inc_A);
    });
}

#define FOREACH_TYPE(T) \
template void set(const communicator& comm, const config& cfg, len_type n, \
                  T alpha, T* A, stride_type inc_A);
#include "configs/foreach_type.h"

}
}
