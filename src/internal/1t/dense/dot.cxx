#include "dot.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

void dot(type_t type, const communicator& comm, const config& cfg,
         const len_vector& len_AB,
         bool conj_A, char* A, const stride_vector& stride_A_AB,
         bool conj_B, char* B, const stride_vector& stride_B_AB,
         char* result)
{
    bool empty = len_AB.size() == 0;

    const len_type ts = type_size[type];

    len_type n0 = (empty ? 1 : len_AB[0]);
    len_vector len1(len_AB.begin() + !empty, len_AB.end());
    len_type n1 = stl_ext::prod(len1);

    stride_type stride_A0 = (empty ? 1 : stride_A_AB[0]);
    stride_type stride_B0 = (empty ? 1 : stride_B_AB[0]);
    len_vector stride_A1, stride_B1;
    for (unsigned i = 1;i < stride_A_AB.size();i++) stride_A1.push_back(stride_A_AB[i]*ts);
    for (unsigned i = 1;i < stride_B_AB.size();i++) stride_B1.push_back(stride_B_AB[i]*ts);

    atomic_accumulator local_result;

    comm.distribute_over_threads(n0, n1,
    [&](len_type n0_min, len_type n0_max, len_type n1_min, len_type n1_max)
    {
        auto A1 = A;
        auto B1 = B;

        viterator<2> iter_AB(len1, stride_A1, stride_B1);
        iter_AB.position(n1_min, A1, B1);
        A1 += n0_min*stride_A0*ts;
        B1 += n0_min*stride_B0*ts;

        scalar micro_result(0, type);

        for (len_type i = n1_min;i < n1_max;i++)
        {
            iter_AB.next(A1, B1);
            cfg.dot_ukr.call(type, n0_max-n0_min, conj_A, A1, stride_A0,
                             conj_B, B1, stride_B0, &micro_result);
        }

        local_result += micro_result;
    });

    reduce(type, comm, local_result);
    if (comm.master()) local_result.store(type, result);

    comm.barrier();
}

}
}
