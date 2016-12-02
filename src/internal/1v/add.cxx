#include "add.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void add(const communicator& comm, const config& cfg, len_type n,
         T alpha, bool conj_A, const T* A, stride_type inc_A,
         T  beta, bool conj_B,       T* B, stride_type inc_B)
{
    len_type n_min, n_max;
    std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(n);

    if (beta == T(0))
    {
        cfg.copy_ukr.call<T>(n_max-n_min,
                             alpha, conj_A, A + n_min*inc_A, inc_A,
                                            B + n_min*inc_B, inc_B);
    }
    else
    {
        cfg.add_ukr.call<T>(n_max-n_min,
                            alpha, conj_A, A + n_min*inc_A, inc_A,
                             beta, conj_B, B + n_min*inc_B, inc_B);
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void add(const communicator& comm, const config& cfg, len_type n, \
                  T alpha, bool conj_A, const T* A, stride_type inc_A, \
                  T  beta, bool conj_B,       T* B, stride_type inc_B);
#include "configs/foreach_type.h"

}
}
