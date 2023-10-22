#include "util.hpp"
#include "dot.hpp"
#include "internal/1t/dense/dot.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dot_full(const communicator& comm, const config& cfg,
              bool conj_A, const dpd_marray_view<T>& A,
              const dim_vector& idx_A_AB,
              bool conj_B, const dpd_marray_view<T>& B,
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
               bool conj_A, const dpd_marray_view<char>& A,
               const dim_vector& idx_A_AB,
               bool conj_B, const dpd_marray_view<char>& B,
               const dim_vector& idx_B_AB,
               char* result)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();
    const auto irrep = A.irrep();
    const auto ndim = A.dimension();

    stride_type nblock_AB = ipow(nirrep, ndim-1);

    std::array<len_vector,1> dense_len;
    std::array<stride_vector,1> dense_stride;
    dense_total_lengths_and_strides(dense_len, dense_stride, A, idx_A_AB);

    stride_type dense_size = stl_ext::prod(dense_len[0]);
    if (nblock_AB > 1)
        dense_size = std::max<stride_type>(1, dense_size/nirrep);

    atomic_accumulator local_result;

    comm.do_tasks_deferred(nblock_AB, dense_size*inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
        {
            tasks.visit(block_AB,
            [&,block_AB](const communicator& subcomm)
            {
                irrep_vector irreps_A(ndim);
                irrep_vector irreps_B(ndim);

                assign_irreps(ndim, irrep, nirrep, block_AB,
                              irreps_A, idx_A_AB, irreps_B, idx_B_AB);

                if (is_block_empty(A, irreps_A)) return;

                marray_view<char> local_A = A(irreps_A);
                marray_view<char> local_B = B(irreps_B);

                auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
                auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
                auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

                scalar block_result(0, type);

                dot(type, subcomm, cfg, len_AB,
                    conj_A, A.data() + (local_A.data()-A.data())*ts, stride_A_AB,
                    conj_B, B.data() + (local_B.data()-B.data())*ts, stride_B_AB,
                    block_result.raw());

                if (subcomm.master()) local_result += block_result;
            });
        }
    });

    reduce(type, comm, local_result);
    if (comm.master()) local_result.store(type, result);
}

void dot(type_t type, const communicator& comm, const config& cfg,
         bool conj_A, const dpd_marray_view<char>& A,
         const dim_vector& idx_A_AB,
         bool conj_B, const dpd_marray_view<char>& B,
         const dim_vector& idx_B_AB,
         char* result)
{
    if (A.irrep() != B.irrep())
    {
        if (comm.master()) memset(result, 0, type_size[type]);
        comm.barrier();
        return;
    }

    if (dpd_impl == FULL)
    {
        switch (type)
        {
            case TYPE_FLOAT:
                dot_full(comm, cfg,
                         conj_A, reinterpret_cast<const dpd_marray_view<float>&>(A), idx_A_AB,
                         conj_B, reinterpret_cast<const dpd_marray_view<float>&>(B), idx_B_AB,
                         result);
                break;
            case TYPE_DOUBLE:
                dot_full(comm, cfg,
                         conj_A, reinterpret_cast<const dpd_marray_view<double>&>(A), idx_A_AB,
                         conj_B, reinterpret_cast<const dpd_marray_view<double>&>(B), idx_B_AB,
                         result);
                break;
            case TYPE_SCOMPLEX:
                dot_full(comm, cfg,
                         conj_A, reinterpret_cast<const dpd_marray_view<scomplex>&>(A), idx_A_AB,
                         conj_B, reinterpret_cast<const dpd_marray_view<scomplex>&>(B), idx_B_AB,
                         result);
                break;
            case TYPE_DCOMPLEX:
                dot_full(comm, cfg,
                         conj_A, reinterpret_cast<const dpd_marray_view<dcomplex>&>(A), idx_A_AB,
                         conj_B, reinterpret_cast<const dpd_marray_view<dcomplex>&>(B), idx_B_AB,
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
