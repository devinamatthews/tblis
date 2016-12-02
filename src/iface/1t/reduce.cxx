#include "reduce.h"

#include "util/macros.h"
#include "util/tensor.hpp"
#include "internal/1t/reduce.hpp"

namespace tblis
{

extern "C"
{

void tblis_tensor_reduce(const tblis_comm* comm, const tblis_config* cfg,
                         reduce_t op, const tblis_tensor* A, const label_type* idx_A_,
                         tblis_scalar* result, len_type* idx)
{
    TBLIS_ASSERT(A->type == result->type);

    unsigned ndim_A = A->ndim;
    std::vector<len_type> len_A;
    std::vector<stride_type> stride_A;
    std::vector<label_type> idx_A;
    diagonal(ndim_A, A->len, A->stride, idx_A_, len_A, stride_A, idx_A);

    fold(len_A, idx_A, stride_A);

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        if (A->alpha<T>() < T(0))
        {
            if (op == REDUCE_MIN) op = REDUCE_MAX;
            else if (op == REDUCE_MAX) op = REDUCE_MIN;
        }

        parallelize_if(internal::reduce<T>, comm, get_config(cfg), op, len_A,
                       static_cast<const T*>(A->data), stride_A, result->get<T>(), *idx);

        if (A->conj)
        {
            result->get<T>() = conj(result->get<T>());
        }

        if (op == REDUCE_SUM || op == REDUCE_SUM_ABS || op == REDUCE_NORM_2)
        {
            result->get<T>() *= A->alpha<T>();
        }
    })
}

}

}
