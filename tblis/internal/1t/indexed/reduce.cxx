#include <tblis/internal/indexed.hpp>
#include <tblis/internal/dpd.hpp>
#include <tblis/internal/thread.hpp>

namespace tblis
{
namespace internal
{

void reduce(type_t type, const communicator& comm, const config& cfg, reduce_t op,
            const indexed_marray_view<char>& A, const dim_vector&,
            char* result, len_type& idx)
{
    const len_type ts = type_size[type];

    scalar local_result(0, type);
    len_type local_idx;
    reduce_init(op, local_result, local_idx);

    for (len_type i = 0;i < A.num_indices();i++)
    {
        scalar block_result(0, type);
        len_type block_idx;

        reduce(type, comm, cfg, op, A.dense_lengths(), A.data(i),
               A.dense_strides(), block_result.raw(), block_idx);

        block_idx += (A.data(i)-A.data(0))/ts;

        switch (type)
        {
            case FLOAT:    block_result.data.s *= reinterpret_cast<const indexed_marray_view<   float>&>(A).factor(i); break;
            case DOUBLE:   block_result.data.d *= reinterpret_cast<const indexed_marray_view<  double>&>(A).factor(i); break;
            case SCOMPLEX: block_result.data.c *= reinterpret_cast<const indexed_marray_view<scomplex>&>(A).factor(i); break;
            case DCOMPLEX: block_result.data.z *= reinterpret_cast<const indexed_marray_view<dcomplex>&>(A).factor(i); break;
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
