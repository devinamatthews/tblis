#include "tblis_transpose.hpp"

#include "external/stl_ext/include/algorithm.hpp"

#include "tblis_config.hpp"
#include "tblis_tensor_detail.hpp"

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_transpose_impl(const std::vector<len_type>& len_AB,
                          T alpha, const T* TBLIS_RESTRICT A, const std::vector<stride_type>& stride_A_AB,
                          T  beta,       T* TBLIS_RESTRICT B, const std::vector<stride_type>& stride_B_AB)
{
    unsigned ndim = len_AB.size();
    auto idx = detail::sort_by_stride(stride_A_AB, stride_B_AB);

    std::vector<stride_type> strides_Ar(ndim-1);
    std::vector<stride_type> strides_Br(ndim-1);
    std::vector<len_type> len(ndim-1);

    for (unsigned i = 0;i < ndim-1;i++)
    {
        strides_Ar[i] = stride_A_AB[idx[i+1]];
        strides_Br[i] = stride_B_AB[idx[i+1]];
        len[i] = len_AB[idx[i+1]];
    }

    stride_type stride_A0 = stride_A_AB[idx[0]];
    stride_type stride_B0 = stride_B_AB[idx[0]];
    len_type len0 = len_AB[idx[0]];

    parallelize
    (
        [&](thread_communicator& comm)
        {
            MArray::viterator<2> iter_AB(len, strides_Ar, strides_Br);

            len_type n = stl_ext::prod(len);
            int nt = comm.num_threads();

            int nt_outer, nt_inner;
            partition_2x2(nt, n, len0, nt_outer, nt_inner);

            thread_communicator subcomm = comm.gang_evenly(nt_outer);

            len_type n_min, n_max;
            std::tie(n_min, n_max, std::ignore) =
                subcomm.distribute_over_gangs(nt_outer, n);

            stride_type off_A, off_B;
            iter_AB.position(n_min, off_A, off_B);

            const T* TBLIS_RESTRICT A_ = A + off_A;
                  T* TBLIS_RESTRICT B_ = B + off_B;

            if (alpha == T(0))
            {
                if (beta == T(0))
                {
                    for (len_type i = n_min;i < n_max;i++)
                    {
                        iter_AB.next(A_, B_);
                        tblis_zerov_ref(subcomm, len0, B_, stride_B0);
                    }
                }
                else if (beta == T(1))
                {
                    // do nothing
                }
                else
                {
                    for (len_type i = n_min;i < n_max;i++)
                    {
                        iter_AB.next(A_, B_);
                        tblis_scalv_ref(subcomm, len0, beta, B_, stride_B0);
                    }
                }
            }
            else if (alpha == T(1))
            {
                if (beta == T(0))
                {
                    for (len_type i = n_min;i < n_max;i++)
                    {
                        iter_AB.next(A_, B_);
                        tblis_copyv_ref(subcomm, false, len0, A_, stride_A0, B_, stride_B0);
                    }
                }
                else if (beta == T(1))
                {
                    for (len_type i = n_min;i < n_max;i++)
                    {
                        iter_AB.next(A_, B_);
                        tblis_addv_ref(subcomm, false, len0, A_, stride_A0, B_, stride_B0);
                    }
                }
                else
                {
                    for (len_type i = n_min;i < n_max;i++)
                    {
                        iter_AB.next(A_, B_);
                        tblis_xpbyv_ref(subcomm, false, len0, A_, stride_A0, beta, B_, stride_B0);
                    }
                }
            }
            else
            {
                if (beta == T(0))
                {
                    for (len_type i = n_min;i < n_max;i++)
                    {
                        iter_AB.next(A_, B_);
                        tblis_scal2v_ref(subcomm, false, len0, alpha, A_, stride_A0, B_, stride_B0);
                    }
                }
                else if (beta == T(1))
                {
                    for (len_type i = n_min;i < n_max;i++)
                    {
                        iter_AB.next(A_, B_);
                        tblis_axpyv_ref(subcomm, false, len0, alpha, A_, stride_A0, B_, stride_B0);
                    }
                }
                else
                {
                    for (len_type i = n_min;i < n_max;i++)
                    {
                        iter_AB.next(A_, B_);
                        tblis_axpbyv_ref(subcomm, false, len0, alpha, A_, stride_A0, beta, B_, stride_B0);
                    }
                }
            }
        }
    );

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_transpose_impl(const std::vector<idx_type>& len_AB, \
                          T alpha, const T* A, const std::vector<stride_type>& stride_A_AB, \
                          T  beta,       T* B, const std::vector<stride_type>& stride_B_AB);
#include "tblis_instantiate_for_types.hpp"

}
}
