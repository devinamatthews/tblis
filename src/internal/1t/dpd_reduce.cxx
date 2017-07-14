#include "dpd_reduce.hpp"
#include "reduce.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dpd_reduce(const communicator& comm, const config& cfg, reduce_t op,
                const dpd_varray_view<const T>& A, const dim_vector& idx_A_A,
                T& result, len_type& idx)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim = A.dimension();

    T local_result, block_result;
    len_type local_idx, block_idx;
    reduce_init(op, local_result, local_idx);

    stride_type nblock = 1;
    for (unsigned i = 0;i < ndim-1;i++) nblock *= nirrep;

    irrep_vector irreps(ndim);
    unsigned irrep = A.irrep();

    for (stride_type block = 0;block < nblock;block++)
    {
        detail::assign_irreps(ndim, irrep, nirrep, block, irreps, idx_A_A);

        if (detail::is_block_empty(A, irreps)) continue;

        auto local_A = A(irreps);

        reduce<T>(comm, cfg, op, local_A.lengths(), local_A.data(),
                  local_A.strides(), block_result, block_idx);
        block_idx += local_A.data() - A.data();

        if (comm.master())
        {
            if (op == REDUCE_SUM || op == REDUCE_SUM_ABS)
            {
                local_result += block_result;
            }
            else if (op == REDUCE_MAX)
            {
                if (block_result > local_result)
                {
                    local_result = block_result;
                    local_idx = block_idx;
                }
            }
            else if (op == REDUCE_MAX_ABS)
            {
                if (std::abs(block_result) > std::abs(local_result))
                {
                    local_result = block_result;
                    local_idx = block_idx;
                }
            }
            else if (op == REDUCE_MIN)
            {
                if (block_result < local_result)
                {
                    local_result = block_result;
                    local_idx = block_idx;
                }
            }
            else if (op == REDUCE_MIN_ABS)
            {
                if (std::abs(block_result) < std::abs(local_result))
                {
                    local_result = block_result;
                    local_idx = block_idx;
                }
            }
            else if (op == REDUCE_NORM_2)
            {
                local_result += block_result*block_result;
            }
        }
    }

    if (comm.master())
    {
        if (op == REDUCE_NORM_2) local_result = sqrt(local_result);
        result = local_result;
        idx = local_idx;
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void dpd_reduce(const communicator& comm, const config& cfg, reduce_t op, \
                         const dpd_varray_view<const T>& A, const dim_vector&, \
                         T& result, len_type& idx);
#include "configs/foreach_type.h"

}
}
