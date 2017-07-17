#include "reduce.hpp"
#include "internal/1t/dpd/reduce.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void reduce(const communicator& comm, const config& cfg, reduce_t op,
            const indexed_dpd_varray_view<const T>& A, const dim_vector& idx_A_A,
            T& result, len_type& idx)
{
    T local_result, block_result;
    len_type local_idx, block_idx;
    reduce_init(op, local_result, local_idx);

    A.for_each_index(
    [&](const dpd_varray_view<const T>& local_A)
    {
        reduce(comm, cfg, op, local_A, idx_A_A, block_result, block_idx);
        block_idx += local_A.data() - A.data(0);

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
template void reduce(const communicator& comm, const config& cfg, reduce_t op, \
                     const indexed_dpd_varray_view<const T>& A, const dim_vector&, \
                     T& result, len_type& idx);
#include "configs/foreach_type.h"

}
}
