#include "tblis.hpp"
#include "impl/tensor_impl.hpp"

using namespace std;
using namespace tblis::blis_like;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_transpose_reference(T alpha, const Tensor<T>& A, const std::string& idx_A,
                               T  beta,       Tensor<T>& B, const std::string& idx_B)
{
    gint_t ndim = A.dimension();

    if (ndim == 0)
    {
        B.data()[0] = alpha*A.data()[0] + beta*B.data()[0];
        return 0;
    }

    const vector<inc_t>& strides_A = A.strides();
    const vector<inc_t>& strides_B = B.strides();
    const vector<dim_t>& len_A = A.lengths();

    string idx;
    for (gint_t i = 0;i < ndim;i++) idx.push_back(i);

    sort(idx.begin(), idx.end(),
    [&](char a, char b)
    {
        return min(strides_A[a], strides_B[a]) <
               min(strides_A[b], strides_B[b]);
    });

    vector<inc_t> strides_Ar(ndim-1);
    vector<inc_t> strides_Br(ndim-1);
    vector<dim_t> len(ndim-1);

    for (gint_t i = 0;i < ndim-1;i++)
    {
        strides_Ar[i] = strides_A[idx[i+1]];
        strides_Br[i] = strides_B[idx[i+1]];
        len[i] = len_A[idx[i+1]];
    }

    inc_t stride_A0 = strides_A[idx[0]];
    inc_t stride_B0 = strides_B[idx[0]];
    dim_t len0 = len_A[idx[0]];

    Iterator<2> iter_AB(len, strides_Ar, strides_Br);

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

template
int tensor_transpose_reference<   float>(   float alpha, const Tensor<   float>& A, const std::string& idx_A,
                                            float  beta,       Tensor<   float>& B, const std::string& idx_B);

template
int tensor_transpose_reference<  double>(  double alpha, const Tensor<  double>& A, const std::string& idx_A,
                                           double  beta,       Tensor<  double>& B, const std::string& idx_B);

template
int tensor_transpose_reference<sComplex>(sComplex alpha, const Tensor<sComplex>& A, const std::string& idx_A,
                                         sComplex  beta,       Tensor<sComplex>& B, const std::string& idx_B);

template
int tensor_transpose_reference<dComplex>(dComplex alpha, const Tensor<dComplex>& A, const std::string& idx_A,
                                         dComplex  beta,       Tensor<dComplex>& B, const std::string& idx_B);

}
}
