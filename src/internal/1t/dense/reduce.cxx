#include "reduce.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

void reduce(type_t type, const communicator& comm, const config& cfg, reduce_t op,
            const len_vector& len_A,
            char* A, const stride_vector& stride_A,
            char* result, len_type& idx)
{
    bool empty = len_A.size() == 0;

    const len_type ts = type_size[type];

    len_type n0 = (empty ? 1 : len_A[0]);
    len_vector len1(len_A.begin() + !empty, len_A.end());
    len_type n1 = stl_ext::prod(len1);

    stride_type stride0 = (empty ? 1 : stride_A[0]);
    len_vector stride1;
    for (auto i : range(1,len_A.size())) stride1.push_back(stride_A[i]*ts);

    atomic_reducer local_result(op);

    comm.distribute_over_threads(n0, n1,
    [&](len_type n0_min, len_type n0_max, len_type n1_min, len_type n1_max)
    {
        auto A1 = A;

        viterator<1> iter_A(len1, stride1);
        iter_A.position(n1_min, A1);

        A1 += n0_min*stride0*ts;

        scalar micro_result(0, type);
        len_type micro_idx;
        reduce_init(op, micro_result, micro_idx);

        for (len_type i = n1_min;i < n1_max;i++)
        {
            auto old_idx = micro_idx;
            micro_idx = -1;

            iter_A.next(A1);
            cfg.reduce_ukr.call(type, op, n0_max-n0_min, A1, stride0, &micro_result, micro_idx);

            if (micro_idx != -1) micro_idx += (A1-A)/ts;
            else micro_idx = old_idx;
        }

        atomic_reduce(op, local_result, micro_result, micro_idx);
    });

    reduce(type, comm, op, local_result);

    if (comm.master()) local_result.store(type, result, idx);

    comm.barrier();
}

}
}
