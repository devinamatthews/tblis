#include "scale.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void scale(const communicator& comm, const config& cfg, len_type n,
           T alpha, bool conj_A, T* A, stride_type inc_A)
{
    comm.distribute_over_threads(n,
    [&](len_type n_min, len_type n_max)
    {
        cfg.scale_ukr.call<T>(n_max-n_min, alpha, conj_A, A+n_min*inc_A, inc_A);
    });
}

#define FOREACH_TYPE(T) \
template void scale(const communicator& comm, const config& cfg, len_type n, \
                    T alpha, bool conj_A, T* A, stride_type inc_A);
#include "configs/foreach_type.h"

}
}
