#include <tblis/internal/indexed.hpp>
#include <tblis/internal/dpd.hpp>
#include <tblis/internal/thread.hpp>

namespace tblis
{
namespace internal
{

template <typename T>
void dot_full(const communicator& comm, const config& cfg,
              bool conj_A, const indexed_marray_view<T>& A,
              const dim_vector& idx_A_AB,
              bool conj_B, const indexed_marray_view<T>& B,
              const dim_vector& idx_B_AB,
              char* result)
{
    marray<T> A2, B2;

    comm.broadcast(
    [&](marray<T>& A2, marray<T>& B2)
    {
        block_to_full(comm, cfg, A, A2);
        block_to_full(comm, cfg, B, B2);

        auto len_AB = stl_ext::select_from(A2.lengths(), idx_A_AB);
        auto stride_A_AB = stl_ext::select_from(A2.strides(), idx_A_AB);
        auto stride_B_AB = stl_ext::select_from(B2.strides(), idx_B_AB);

        dot(type_tag<T>::value, comm, cfg, len_AB,
            conj_A, reinterpret_cast<char*>(A2.data()), stride_A_AB,
            conj_B, reinterpret_cast<char*>(B2.data()), stride_B_AB,
            result);
    },
    A2, B2);
}

void dot_block(type_t type, const communicator& comm, const config& cfg,
               bool conj_A, const indexed_marray_view<char>& A,
               const dim_vector& idx_A_AB,
               bool conj_B, const indexed_marray_view<char>& B,
               const dim_vector& idx_B_AB,
               char* result)
{
    const len_type ts = type_size[type];

    index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);

    group_indices<1> indices_A(type, A, group_AB, 0);
    group_indices<1> indices_B(type, B, group_AB, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();

    atomic_accumulator local_result;

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    comm.do_tasks_deferred(std::min(nidx_A, nidx_B),
                           stl_ext::prod(group_AB.dense_len)*inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        for_each_match<false, false>(idx_A, nidx_A, indices_A, 0,
                                     idx_B, nidx_B, indices_B, 0,
        [&]
        {
            auto factor = indices_A[idx_A].factor*indices_B[idx_B].factor;
            if (factor.is_zero()) return;

            tasks.visit(idx++,
            [&,idx_A,idx_B,factor](const communicator& subcomm)
            {
                stride_type off_A_AB, off_B_AB;
                get_local_offset(indices_A[idx_A].idx[0], group_AB,
                                 off_A_AB, 0, off_B_AB, 1);

                auto data_A = A.data(0) + (indices_A[idx_A].offset + off_A_AB)*ts;
                auto data_B = B.data(0) + (indices_B[idx_B].offset + off_B_AB)*ts;

                scalar block_result(0, type);

                dot(type, subcomm, cfg, group_AB.dense_len,
                    conj_A, data_A, group_AB.dense_stride[0],
                    conj_B, data_B, group_AB.dense_stride[1],
                    block_result.raw());

                if (subcomm.master()) local_result += factor*block_result;
            });
        });
    });

    reduce(type, comm, local_result);
    if (comm.master()) local_result.store(type, result);
}

void dot(type_t type, const communicator& comm, const config& cfg,
         bool conj_A, const indexed_marray_view<char>& A,
         const dim_vector& idx_A_AB,
         bool conj_B, const indexed_marray_view<char>& B,
         const dim_vector& idx_B_AB,
         char* result)
{
    if (dpd_impl == FULL)
    {
        switch (type)
        {
            case FLOAT:
                dot_full(comm, cfg,
                         conj_A, reinterpret_cast<const indexed_marray_view<float>&>(A), idx_A_AB,
                         conj_B, reinterpret_cast<const indexed_marray_view<float>&>(B), idx_B_AB,
                         result);
                break;
            case DOUBLE:
                dot_full(comm, cfg,
                         conj_A, reinterpret_cast<const indexed_marray_view<double>&>(A), idx_A_AB,
                         conj_B, reinterpret_cast<const indexed_marray_view<double>&>(B), idx_B_AB,
                         result);
                break;
            case SCOMPLEX:
                dot_full(comm, cfg,
                         conj_A, reinterpret_cast<const indexed_marray_view<scomplex>&>(A), idx_A_AB,
                         conj_B, reinterpret_cast<const indexed_marray_view<scomplex>&>(B), idx_B_AB,
                         result);
                break;
            case DCOMPLEX:
                dot_full(comm, cfg,
                         conj_A, reinterpret_cast<const indexed_marray_view<dcomplex>&>(A), idx_A_AB,
                         conj_B, reinterpret_cast<const indexed_marray_view<dcomplex>&>(B), idx_B_AB,
                         result);
                break;
        }
    }
    else
    {
        dot_block(type, comm, cfg,
                  conj_A, A, idx_A_AB,
                  conj_B, B, idx_B_AB,
                  result);
    }

    comm.barrier();
}

}
}
