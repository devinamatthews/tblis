#include "dpd_set.hpp"
#include "set.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dpd_set(const communicator& comm, const config& cfg,
             T alpha, const dpd_varray_view<T>& A, const std::vector<unsigned>&)
{
    A.for_each_block(
    [&](const varray_view<T>& A2)
    {
        set(comm, cfg, A2.lengths(), alpha, A2.data(), A2.strides());
    });
}

#define FOREACH_TYPE(T) \
template void dpd_set(const communicator& comm, const config& cfg, \
                      T alpha, const dpd_varray_view<T>& A, const std::vector<unsigned>&);
#include "configs/foreach_type.h"

}
}
