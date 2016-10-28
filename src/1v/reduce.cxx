#include "reduce.h"

#include "configs/configs.hpp"

namespace tblis
{

void reduce_int(const communicator& comm, const config& cfg,
                reduce_t op, const tblis_vector& A,
                tblis_scalar& result, len_type& idx)
{
    TBLIS_ASSERT(A.type == result.type);

    TBLIS_WITH_TYPE_AS(A.type, T,
    {
        typedef std::numeric_limits<real_type_t<T>> limits;

        switch (op)
        {
            case REDUCE_SUM:
            case REDUCE_SUM_ABS:
            case REDUCE_MAX_ABS:
            case REDUCE_NORM_2:
                result.get<T>() = T();
                break;
            case REDUCE_MAX:
                result.get<T>() = limits::min();
                break;
            case REDUCE_MIN:
            case REDUCE_MIN_ABS:
                result.get<T>() = limits::max();
                break;
        }

        idx = -1;

        cfg.reduce_ukr.call<T>(comm, op, A.n,
            (const T*)A.data, A.inc, result.get<T>(), idx);

        reduce(comm, op, result.get<T>(), idx);
    })
}

extern "C"
{

void tblis_vector_reduce(const tblis_comm* comm, const tblis_config* cfg,
                         reduce_t op, const tblis_vector* A,
                         tblis_scalar* result, len_type* idx)
{
    parallelize_if(reduce_int, comm, get_config(cfg),
                   op, *A, *result, *idx);
}

}

}
