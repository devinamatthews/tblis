#include "util.hpp"
#include "reduce.hpp"
#include "internal/0/reduce.hpp"
#include "internal/1t/dense/reduce.hpp"

namespace tblis
{
namespace internal
{

void reduce(type_t type, const communicator& comm, const config& cfg, reduce_t op,
            const dpd_varray_view<char>& A, const dim_vector& idx_A,
            char* result, len_type& idx)
{
    const len_type ts = type_size[type];

    const unsigned nirrep = A.num_irreps();
    const unsigned irrep = A.irrep();
    const unsigned ndim = A.dimension();

    scalar local_result(0, type);
    len_type local_idx;
    reduce_init(op, local_result, local_idx);

    stride_type nblock = 1;
    for (unsigned i = 0;i < ndim-1;i++) nblock *= nirrep;

    irrep_vector irreps(ndim);

    for (stride_type block = 0;block < nblock;block++)
    {
        assign_irreps(ndim, irrep, nirrep, block, irreps, idx_A);

        if (is_block_empty(A, irreps)) continue;

        varray_view<char> local_A = A(irreps);

        scalar block_result(0, type);
        len_type block_idx;

        reduce(type, comm, cfg, op, local_A.lengths(),
               A.data() + (local_A.data()-A.data())*ts,
               local_A.strides(), block_result.raw(), block_idx);

        block_idx += local_A.data()-A.data();

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
