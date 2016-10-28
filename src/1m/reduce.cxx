#include "reduce.h"

#include "configs/configs.hpp"

namespace tblis
{

extern "C"
{

void tblis_matrix_reduce_int(const communicator& comm, const config& cfg,
                             reduce_t op, const tblis_matrix& A,
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

        if (A.rs < A.cs)
        {
            for (len_type j = 0;j < A.n;j++)
            {
                int old_idx = idx;
                idx = -1;

                cfg.reduce_ukr.call<T>(comm, op, A.m,
                    (const T*)A.data + j*A.cs, A.rs, result.get<T>(), idx);

                if (idx != -1) idx += j*A.cs;
                else idx = old_idx;
            }
        }
        else
        {
            for (len_type i = 0;i < A.m;i++)
            {
                int old_idx = idx;
                idx = -1;

                cfg.reduce_ukr.call<T>(comm, op, A.n,
                    (const T*)A.data + i*A.rs, A.cs, result.get<T>(), idx);

                if (idx != -1) idx += i*A.rs;
                else idx = old_idx;
            }
        }

        reduce(comm, op, result.get<T>(), idx);
    })
}

void tblis_matrix_reduce(const tblis_comm* comm, const tblis_config* cfg,
                         reduce_t op, const tblis_matrix* A,
                         tblis_scalar* result, len_type* idx)
{
    parallelize_if(tblis_matrix_reduce_int, comm, get_config(cfg),
                   op, *A, *result, *idx);
}

}

}
