#include "reduce.hpp"
#include "internal/1t/dense/reduce.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void reduce(const communicator& comm, const config& cfg, reduce_t op,
            const indexed_varray_view<const T>& A, const dim_vector& idx_A_A,
            T& result, len_type& idx)
{
    T local_result, block_result;
    len_type local_idx, block_idx;
    reduce_init(op, local_result, local_idx);

    for (len_type i = 0;i < A.num_indices();i++)
    {
        auto factor = A.factor(i);

        reduce(comm, cfg, op, A.dense_lengths(), A.data(i),
               A.dense_strides(), block_result, block_idx);
        block_idx += A.data(i) - A.data(0);

        if (comm.master())
        {
            if (op == REDUCE_SUM || op == REDUCE_SUM_ABS)
            {
                local_result += factor*block_result;
            }
            else if (op == REDUCE_MAX)
            {
                if (factor*block_result > local_result)
                {
                    local_result = factor*block_result;
                    local_idx = block_idx;
                }
            }
            else if (op == REDUCE_MAX_ABS)
            {
                if (std::abs(factor*block_result) > local_result)
                {
                    local_result = std::abs(factor*block_result);
                    local_idx = block_idx;
                }
            }
            else if (op == REDUCE_MIN)
            {
                if (factor*block_result < local_result)
                {
                    local_result = factor*block_result;
                    local_idx = block_idx;
                }
            }
            else if (op == REDUCE_MIN_ABS)
            {
                if (std::abs(factor*block_result) < local_result)
                {
                    local_result = std::abs(factor*block_result);
                    local_idx = block_idx;
                }
            }
            else if (op == REDUCE_NORM_2)
            {
                local_result += factor*factor*block_result*block_result;
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
template void reduce(const communicator& comm, const config& cfg, reduce_t op, \
                     const indexed_varray_view<const T>& A, const dim_vector&, \
                     T& result, len_type& idx);
#include "configs/foreach_type.h"

}
}
