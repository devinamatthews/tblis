#include "tblis.hpp"

using namespace std;
using namespace MArray;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_transpose_reference(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                               T  beta, const       tensor_view<T>& B, const std::string& idx_B)
{
    unsigned ndim = A.dimension();

    if (ndim == 0)
    {
        B.data()[0] = alpha*A.data()[0] + beta*B.data()[0];
        return 0;
    }

    const vector<stride_type>& strides_A = A.strides();
    vector<stride_type> strides_B(ndim);
    const vector<idx_type>& len_A = A.lengths();

    for (unsigned i = 0;i < ndim;i++)
        for (unsigned j = 0;j < ndim;j++)
            if (idx_A[i] == idx_B[j]) strides_B[i] = B.stride(j);

    string idx;
    for (unsigned i = 0;i < ndim;i++) idx.push_back(i);

    sort(idx.begin(), idx.end(),
    [&](char a, char b)
    {
        return min(strides_A[a], strides_B[a]) <
               min(strides_A[b], strides_B[b]);
    });

    vector<stride_type> strides_Ar(ndim-1);
    vector<stride_type> strides_Br(ndim-1);
    vector<idx_type> len(ndim-1);

    for (unsigned i = 0;i < ndim-1;i++)
    {
        strides_Ar[i] = strides_A[idx[i+1]];
        strides_Br[i] = strides_B[idx[i+1]];
        len[i] = len_A[idx[i+1]];
    }

    stride_type stride_A0 = strides_A[idx[0]];
    stride_type stride_B0 = strides_B[idx[0]];
    idx_type len0 = len_A[idx[0]];

    parallelize
    (
        [&](thread_communicator& comm)
        {
            viterator<2> iter_AB(len, strides_Ar, strides_Br);

            idx_type n = stl_ext::prod(len);
            int nt = comm.num_threads();

            int nt_outer, nt_inner;
            partition_2x2(nt, n, len0, nt_outer, nt_inner);

            thread_communicator subcomm = comm.gang_evenly(nt_outer);

            idx_type n_min, n_max;
            std::tie(n_min, n_max, std::ignore) =
                subcomm.distribute_over_gangs(nt_outer, n);

            stride_type off_A, off_B;
            iter_AB.position(n_min, off_A, off_B);

            const T* A_ = A.data() + off_A;
                  T* B_ = B.data() + off_B;

            if (alpha == T(0))
            {
                if (beta == T(0))
                {
                    for (idx_type i = n_min;i < n_max;i++)
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
                    for (idx_type i = n_min;i < n_max;i++)
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
                    for (idx_type i = n_min;i < n_max;i++)
                    {
                        iter_AB.next(A_, B_);
                        tblis_copyv_ref(subcomm, false, len0, A_, stride_A0, B_, stride_B0);
                    }
                }
                else if (beta == T(1))
                {
                    for (idx_type i = n_min;i < n_max;i++)
                    {
                        iter_AB.next(A_, B_);
                        tblis_addv_ref(subcomm, false, len0, A_, stride_A0, B_, stride_B0);
                    }
                }
                else
                {
                    for (idx_type i = n_min;i < n_max;i++)
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
                    for (idx_type i = n_min;i < n_max;i++)
                    {
                        iter_AB.next(A_, B_);
                        tblis_scal2v_ref(subcomm, false, len0, alpha, A_, stride_A0, B_, stride_B0);
                    }
                }
                else if (beta == T(1))
                {
                    for (idx_type i = n_min;i < n_max;i++)
                    {
                        iter_AB.next(A_, B_);
                        tblis_axpyv_ref(subcomm, false, len0, alpha, A_, stride_A0, B_, stride_B0);
                    }
                }
                else
                {
                    for (idx_type i = n_min;i < n_max;i++)
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
int tensor_transpose_reference<T>(T alpha, const const_tensor_view<T>& A, const std::string& idx_A, \
                                  T  beta, const       tensor_view<T>& B, const std::string& idx_B);
#include "tblis_instantiate_for_types.hpp"

}
}
