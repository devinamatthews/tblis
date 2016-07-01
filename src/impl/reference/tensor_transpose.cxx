#include "tblis.hpp"
#include "impl/tensor_impl.hpp"

using namespace std;
using namespace MArray;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_transpose_reference(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                               T  beta,             tensor_view<T>& B, const std::string& idx_B)
{
    unsigned ndim = A.dimension();

    if (ndim == 0)
    {
        B.data()[0] = alpha*A.data()[0] + beta*B.data()[0];
        return 0;
    }

    const vector<stride_type>& strides_A = A.strides();
    const vector<stride_type>& strides_B = B.strides();
    const vector<idx_type>& len_A = A.lengths();

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

    viterator<2> iter_AB(len, strides_Ar, strides_Br);

    const T* restrict A_ = A.data();
          T* restrict B_ = B.data();

    if (alpha == 0.0)
    {
        if (beta == 0.0)
        {
            while (iter_AB.next(A_, B_))
            {
                assert (B_-B.data() >= 0 && B_-B.data() < B.size());
                tblis_zerov(len0, B_, stride_B0);
            }
        }
        else if (beta == 1.0)
        {
            // do nothing
        }
        else
        {
            while (iter_AB.next(A_, B_))
            {
                assert (B_-B.data() >= 0 && B_-B.data() < B.size());
                tblis_scalv(len0, beta, B_, stride_B0);
            }
        }
    }
    else if (alpha == 1.0)
    {
        if (beta == 0.0)
        {
            while (iter_AB.next(A_, B_))
            {
                assert (A_-A.data() >= 0 && A_-A.data() < A.size());
                assert (B_-B.data() >= 0 && B_-B.data() < B.size());
                tblis_copyv(false, len0, A_, stride_A0, B_, stride_B0);
            }
        }
        else if (beta == 1.0)
        {
            while (iter_AB.next(A_, B_))
            {
                assert (A_-A.data() >= 0 && A_-A.data() < A.size());
                assert (B_-B.data() >= 0 && B_-B.data() < B.size());
                tblis_addv(false, len0, A_, stride_A0, B_, stride_B0);
            }
        }
        else
        {
            while (iter_AB.next(A_, B_))
            {
                assert (A_-A.data() >= 0 && A_-A.data() < A.size());
                assert (B_-B.data() >= 0 && B_-B.data() < B.size());
                tblis_xpbyv(false, len0, A_, stride_A0, beta, B_, stride_B0);
            }
        }
    }
    else
    {
        if (beta == 0.0)
        {
            while (iter_AB.next(A_, B_))
            {
                assert (A_-A.data() >= 0 && A_-A.data() < A.size());
                assert (B_-B.data() >= 0 && B_-B.data() < B.size());
                tblis_scal2v(false, len0, alpha, A_, stride_A0, B_, stride_B0);
            }
        }
        else if (beta == 1.0)
        {
            while (iter_AB.next(A_, B_))
            {
                assert (A_-A.data() >= 0 && A_-A.data() < A.size());
                assert (B_-B.data() >= 0 && B_-B.data() < B.size());
                tblis_axpyv(false, len0, alpha, A_, stride_A0, B_, stride_B0);
            }
        }
        else
        {
            while (iter_AB.next(A_, B_))
            {
                assert (A_-A.data() >= 0 && A_-A.data() < A.size());
                assert (B_-B.data() >= 0 && B_-B.data() < B.size());
                tblis_axpbyv(false, len0, alpha, A_, stride_A0, beta, B_, stride_B0);
            }
        }
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_transpose_reference<T>(T alpha, const const_tensor_view<T>& A, const std::string& idx_A, \
                                  T  beta,             tensor_view<T>& B, const std::string& idx_B);
#include "tblis_instantiate_for_types.hpp"

}
}
