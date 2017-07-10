#include "dpd_reduce.hpp"
#include "reduce.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dpd_reduce(const communicator& comm, const config& cfg, reduce_t op,
                const dpd_varray_view<const T>& A, const std::vector<unsigned>&,
                T& result, len_type& idx)
{
    T local_result;
    len_type local_idx;
    reduce_init(op, local_result, local_idx);

    A.for_each_block(
    [&](const varray_view<const T>& A2)
    {
        T block_result;
        len_type block_idx;
        reduce(comm, cfg, op, A2.lengths(), A2.data(), A2.strides(),
               block_result, block_idx);

        if (op == REDUCE_SUM || op == REDUCE_SUM_ABS)
        {
            local_result += block_result;
        }
        else if (op == REDUCE_MAX)
        {
            if (block_result > local_result)
            {
                local_result = block_result;
                local_idx = block_idx + (A2.data() - A.data());
            }
        }
        else if (op == REDUCE_MAX_ABS)
        {
            if (std::abs(block_result) > std::abs(local_result))
            {
                local_result = block_result;
                local_idx = block_idx + (A2.data() - A.data());
            }
        }
        else if (op == REDUCE_MIN)
        {
            if (block_result < local_result)
            {
                local_result = block_result;
                local_idx = block_idx + (A2.data() - A.data());
            }
        }
        else if (op == REDUCE_MIN_ABS)
        {
            if (std::abs(block_result) < std::abs(local_result))
            {
                local_result = block_result;
                local_idx = block_idx + (A2.data() - A.data());
            }
        }
        else if (op == REDUCE_NORM_2)
        {
            local_result += block_result*block_result;
        }
    });

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
                         const dpd_varray_view<const T>& A, const std::vector<unsigned>&, \
                         T& result, len_type& idx);
#include "configs/foreach_type.h"

}
}
