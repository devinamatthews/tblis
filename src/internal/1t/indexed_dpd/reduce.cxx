#include "reduce.hpp"
#include "internal/0/reduce.hpp"
#include "internal/1t/dpd/reduce.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

void reduce(type_t type, const communicator& comm, const config& cfg, reduce_t op,
            const indexed_dpd_varray_view<char>& A, const dim_vector& idx_A_A,
            char* result, len_type& idx)
{
    const len_type ts = type_size[type];

    scalar local_result(0, type);
    len_type local_idx;
    reduce_init(op, local_result, local_idx);

    auto local_A = A[0];

    for (len_type i = 0;i < A.num_indices();i++)
    {
        local_A.data(A.data(i));

        scalar block_result(0, type);
        len_type block_idx;

        reduce(type, comm, cfg, op, local_A, idx_A_A, block_result.raw(), block_idx);

        block_idx += (local_A.data()-A.data(0))/ts;

        switch (type)
        {
            case TYPE_FLOAT:    block_result.data.s *= reinterpret_cast<const indexed_dpd_varray_view<   float>&>(A).factor(i); break;
            case TYPE_DOUBLE:   block_result.data.d *= reinterpret_cast<const indexed_dpd_varray_view<  double>&>(A).factor(i); break;
            case TYPE_SCOMPLEX: block_result.data.c *= reinterpret_cast<const indexed_dpd_varray_view<scomplex>&>(A).factor(i); break;
            case TYPE_DCOMPLEX: block_result.data.z *= reinterpret_cast<const indexed_dpd_varray_view<dcomplex>&>(A).factor(i); break;
        }

        reduce(type, op, block_result.raw(), block_idx, local_result.raw(), local_idx);
    }

    if (comm.master())
    {
        if (op == REDUCE_NORM_2) local_result.sqrt();
        local_result.to(result);
        idx = local_idx;
    }

    comm.barrier();
}

}
}
