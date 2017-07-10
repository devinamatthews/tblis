#include "dot.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dpd_dot(const communicator& comm, const config& cfg,
             bool conj_A, const dpd_varray_view<const T>& A,
             const std::vector<unsigned>& idx_A_A,
             const std::vector<unsigned>& idx_A_AB,
             bool conj_B, const dpd_varray_view<const T>& B,
             const std::vector<unsigned>& idx_B_B,
             const std::vector<unsigned>& idx_B_AB,
             T& result)
{
    TBLIS_ASSERT(idx_A_A.empty());
    TBLIS_ASSERT(idx_B_B.empty());

    T local_result = T();

    auto perm = stl_ext::permuted(idx_A_AB, detail::inverse_permutation(idx_B_AB));

    A.for_each_block(
    [&](const varray_view<const T>& A2, const std::vector<unsigned>& irreps_A)
    {
        auto irreps_B = stl_ext::permuted(irreps_A, perm);
        auto B2 = B(irreps_B);
        T block_result;
        dot(comm, cfg, {}, {}, B2.lengths(),
            conj_A, A2.data(), {}, stl_ext::permuted(A2.strides(), perm),
            conj_B, B2.data(), {}, B2.strides(), block_result);
        local_result += block_result;
    });

    if (comm.master()) result = local_result;

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void dpd_dot(const communicator& comm, const config& cfg, \
                      bool conj_A, const dpd_varray_view<const T>& A, \
                      const std::vector<unsigned>& idx_A_A, \
                      const std::vector<unsigned>& idx_A_AB, \
                      bool conj_B, const dpd_varray_view<const T>& B, \
                      const std::vector<unsigned>& idx_B_B, \
                      const std::vector<unsigned>& idx_B_AB, \
                      T& result);
#include "configs/foreach_type.h"

}
}
