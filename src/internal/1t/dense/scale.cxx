#include "scale.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

void scale(type_t type, const communicator& comm, const config& cfg,
           const len_vector& len_A, const scalar& alpha,
           bool conj_A, char* A, const stride_vector& stride_A)
{
    bool empty = len_A.size() == 0;

    const len_type ts = type_size[type];

    len_type n0 = (empty ? 1 : len_A[0]);
    len_vector len1(len_A.begin() + !empty, len_A.end());
    len_type n1 = stl_ext::prod(len1);

    stride_type stride0 = (empty ? 1 : stride_A[0]);
    len_vector stride1;
    for (unsigned i = 1;i < stride_A.size();i++) stride1.push_back(stride_A[i]*ts);

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
            cfg.scale_ukr.call(type, n0_max-n0_min,
                               &alpha, conj_A, A1, stride0);
        }
    });

    comm.barrier();
}

}
}
