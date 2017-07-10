#include "dpd_scale.hpp"
#include "scale.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dpd_scale(const communicator& comm, const config& cfg,
               T alpha, bool conj_A, const dpd_varray_view<T>& A,
               const std::vector<unsigned>&)
{
    A.for_each_block(
    [&](const varray_view<T>& A2)
    {
        scale(comm, cfg, A2.lengths(), alpha, conj_A, A2.data(), A2.strides());
    });
}

#define FOREACH_TYPE(T) \
template void dpd_scale(const communicator& comm, const config& cfg, \
                        T alpha, bool conj_A, const dpd_varray_view<T>& A, \
                        const std::vector<unsigned>&);
#include "configs/foreach_type.h"

}
}
