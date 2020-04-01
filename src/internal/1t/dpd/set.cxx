#include "util.hpp"
#include "set.hpp"
#include "internal/1t/dense/set.hpp"

namespace tblis
{
namespace internal
{

void set(type_t type, const communicator& comm, const config& cfg,
         const scalar& alpha, const dpd_varray_view<char>& A, const dim_vector& idx_A)
{
    const len_type ts = type_size[type];

    const unsigned nirrep = A.num_irreps();
    const unsigned irrep = A.irrep();
    const unsigned ndim = A.dimension();

    stride_type nblock = 1;
    for (unsigned i = 0;i < ndim-1;i++) nblock *= nirrep;

    irrep_vector irreps(ndim);

    for (stride_type block = 0;block < nblock;block++)
    {
        assign_irreps(ndim, irrep, nirrep, block, irreps, idx_A);

        if (is_block_empty(A, irreps)) continue;

        varray_view<char> local_A = A(irreps);

        set(type, comm, cfg, local_A.lengths(), alpha, A.data() + (local_A.data()-A.data())*ts, local_A.strides());
    }
}

}
}
