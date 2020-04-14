#include "set.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

void set(type_t type, const communicator& comm, const config& cfg,
         const len_vector& len_A,
         const scalar& alpha, char* A, const stride_vector& stride_A)
{
    if (len_A.size() == 0)
    {
        alpha.to(A);
        comm.barrier();
        return;
    }

    const len_type ts = type_size[type];

    len_type n0 = len_A[0];
    len_vector len1(len_A.begin() + 1, len_A.end());
    len_type n1 = stl_ext::prod(len1);

    stride_type stride0 = stride_A[0];
    len_vector stride1;
    for (auto i : range(1,len_A.size())) stride1.push_back(stride_A[i]*ts);

    comm.distribute_over_threads(n0, n1,
    [&](len_type n0_min, len_type n0_max, len_type n1_min, len_type n1_max)
    {
        auto A1 = A;

        viterator<1> iter_A(len1, stride1);
        iter_A.position(n1_min, A1);
        A1 += n0_min*stride0*ts;

        for (len_type i = n1_min;i < n1_max;i++)
        {
            iter_A.next(A1);
            cfg.set_ukr.call(type, n0_max-n0_min, &alpha, A1, stride0);
        }
    });

    comm.barrier();
}

}
}
