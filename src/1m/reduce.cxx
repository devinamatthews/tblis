#include "reduce.h"

#include "configs/configs.hpp"
#include "util/assert.h"

namespace tblis
{

extern "C"
{

void tblis_matrix_reduce(len_type m, len_type n,
                         const void* A, type_t type_A,
                         stride_type rs_A, stride_type cs_A,
                         void* norm, type_t type_norm)
{
    parallelize
    (
        [=](communicator& comm)
        {
            tblis_matrix_reduce_coll(comm, m, n, A, type_A, rs_A, cs_A,
                                     norm, type_norm);
        }
    );
}

void tblis_matrix_reduce_single(len_type m, len_type n,
                                const void* A, type_t type_A,
                                stride_type rs_A, stride_type cs_A,
                                void* norm, type_t type_norm)
{
    communicator comm;
    tblis_matrix_reduce_coll(comm, m, n, A, type_A, rs_A, cs_A,
                             norm, type_norm);
}

void tblis_matrix_reduce_coll(tci_comm_t* comm,
                              len_type m, len_type n,
                              const void* A, type_t type_A,
                              stride_type rs_A, stride_type cs_A,
                              void* norm, type_t type_norm)
{
    TBLIS_ASSERT(type_A == type_norm);

    if (type_A == TYPE_SINGLE)
    {
        //TODO
    }
    else if (type_A == TYPE_DOUBLE)
    {
        //TODO
    }
    else if (type_A == TYPE_SCOMPLEX)
    {
        //TODO
    }
    else if (type_A == TYPE_DCOMPLEX)
    {
        //TODO
    }
    else
    {
        TBLIS_ASSERT(0);
    }
}

}

}
