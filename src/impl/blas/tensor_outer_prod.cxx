#include "impl/tensor_impl.hpp"
#include "core/tensor_slicer.hpp"

using namespace std;
using namespace blis;
using namespace tensor::util;

namespace tensor
{
namespace impl
{

template <typename T>
int tensor_outer_prod_blas(T alpha, const Tensor<T>& A, const std::string& idx_A,
                                    const Tensor<T>& B, const std::string& idx_B,
                           T  beta,       Tensor<T>& C, const std::string& idx_C)
{
    string idx_AC_BC(C.getDimension(), 0);

    gint_t ndim_AC =
        set_intersection(idx_A.begin(), idx_A.end(),
                         idx_C.begin(), idx_C.end(),
                         idx_AC_BC.begin()) - idx_AC_BC.begin();

    gint_t ndim_BC =
        set_intersection(idx_B.begin(), idx_B.end(),
                         idx_C.begin(), idx_C.end(),
                         idx_AC_BC.begin()+ndim_AC) - (idx_AC_BC.begin()+ndim_AC);

    assert(ndim_AC+ndim_BC == C.getDimension());

    vector<dim_t> len_AC_BC(ndim_AC+ndim_BC);

    gint_t j = 0;
    for (gint_t i = 0;i < C.getDimension();i++)
    {
        if (ndim_AC+ndim_BC > j && idx_C[i] == idx_AC_BC[j])
        {
            len_AC_BC[j++] = C.getLength(i);
        }
    }
    for (gint_t i = 0;i < C.getDimension();i++)
    {
        if (ndim_AC+ndim_BC > j && idx_C[i] == idx_AC_BC[j])
        {
            len_AC_BC[j++] = C.getLength(i);
        }
        if (i == C.getDimension()-1)
        {
            assert(j == ndim_AC+ndim_BC);
        }
    }

    Tensor<T> ar(A.getDimension(), A.getLengths());
    Tensor<T> br(B.getDimension(), B.getLengths());
    Tensor<T> cr(ndim_AC+ndim_BC, len_AC_BC);

    Matrix<T> am, bm, cm;
    Scalar<T> alp(alpha);
    Scalar<T> zero;

    Matricize(ar, am, 0);
    Matricize(br, bm, 0);
    Matricize(cr, cm, ndim_AC);
    am.setTrans(BLIS_TRANSPOSE);

    Normalize(cr, idx_AC_BC);

    tensor_transpose_impl<T>(1.0, A, idx_A, 0.0, ar, idx_A);
    tensor_transpose_impl<T>(1.0, B, idx_B, 0.0, br, idx_B);
    bli_setm(zero, cm);
    bli_ger(alp, am, bm, cm);
    tensor_transpose_impl<T>(1.0, cr, idx_AC_BC, beta, C, idx_C);

    return 0;
}

template
int tensor_outer_prod_blas<   float>(   float alpha, const Tensor<   float>& A, const std::string& idx_A,
                                                     const Tensor<   float>& B, const std::string& idx_B,
                                        float  beta,       Tensor<   float>& C, const std::string& idx_C);

template
int tensor_outer_prod_blas<  double>(  double alpha, const Tensor<  double>& A, const std::string& idx_A,
                                                     const Tensor<  double>& B, const std::string& idx_B,
                                       double  beta,       Tensor<  double>& C, const std::string& idx_C);

template
int tensor_outer_prod_blas<sComplex>(sComplex alpha, const Tensor<sComplex>& A, const std::string& idx_A,
                                                     const Tensor<sComplex>& B, const std::string& idx_B,
                                     sComplex  beta,       Tensor<sComplex>& C, const std::string& idx_C);

template
int tensor_outer_prod_blas<dComplex>(dComplex alpha, const Tensor<dComplex>& A, const std::string& idx_A,
                                                     const Tensor<dComplex>& B, const std::string& idx_B,
                                     dComplex  beta,       Tensor<dComplex>& C, const std::string& idx_C);

}
}
