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
int tensor_contract_blas(T alpha, const Tensor<T>& A, const std::string& idx_A,
                                  const Tensor<T>& B, const std::string& idx_B,
                         T  beta,       Tensor<T>& C, const std::string& idx_C)
{
    string idx_AB_AC(A.getDimension(), 0);
    string idx_AB_BC(B.getDimension(), 0);
    string idx_AC_BC(C.getDimension(), 0);

    gint_t ndim_AB =
        set_intersection(idx_A.begin(), idx_A.end(),
                         idx_B.begin(), idx_B.end(),
                         idx_AB_AC.begin()) - idx_AB_AC.begin();

    gint_t ndim_AC =
        set_intersection(idx_A.begin(), idx_A.end(),
                         idx_C.begin(), idx_C.end(),
                         idx_AC_BC.begin()) - idx_AC_BC.begin();

    gint_t ndim_BC =
        set_intersection(idx_B.begin(), idx_B.end(),
                         idx_C.begin(), idx_C.end(),
                         idx_AC_BC.begin()+ndim_AC) - (idx_AC_BC.begin()+ndim_AC);

    copy(idx_AB_AC.begin()        , idx_AB_AC.begin()+ndim_AB        , idx_AB_BC.begin()        );
    copy(idx_AC_BC.begin()        , idx_AC_BC.begin()+ndim_AC        , idx_AB_AC.begin()+ndim_AB);
    copy(idx_AC_BC.begin()+ndim_AC, idx_AC_BC.begin()+ndim_AC+ndim_BC, idx_AB_BC.begin()+ndim_AB);

    idx_AB_AC.resize(ndim_AB+ndim_AC);
    idx_AB_BC.resize(ndim_AB+ndim_BC);
    idx_AC_BC.resize(ndim_AC+ndim_BC);

    assert(ndim_AB+ndim_AC == A.getDimension());
    assert(ndim_AB+ndim_BC == B.getDimension());
    assert(ndim_AC+ndim_BC == C.getDimension());

    vector<dim_t> len_AB_AC(ndim_AB+ndim_AC);
    vector<dim_t> len_AB_BC(ndim_AB+ndim_BC);
    vector<dim_t> len_AC_BC(ndim_AC+ndim_BC);

    gint_t j = 0;
    for (gint_t i = 0;i < A.getDimension();i++)
    {
        if (ndim_AB+ndim_AC > j && idx_A[i] == idx_AB_AC[j])
        {
            len_AB_AC[j++] = A.getLength(i);
        }
    }
    for (gint_t i = 0;i < A.getDimension();i++)
    {
        if (ndim_AB+ndim_AC > j && idx_A[i] == idx_AB_AC[j])
        {
            len_AB_AC[j++] = A.getLength(i);
        }
        if (i == A.getDimension()-1)
        {
            assert(j == ndim_AB+ndim_AC);
        }
    }

    j = 0;
    for (gint_t i = 0;i < B.getDimension();i++)
    {
        if (ndim_AB+ndim_BC > j && idx_B[i] == idx_AB_BC[j])
        {
            len_AB_BC[j++] = B.getLength(i);
        }
    }
    for (gint_t i = 0;i < B.getDimension();i++)
    {
        if (ndim_AB+ndim_BC > j && idx_B[i] == idx_AB_BC[j])
        {
            len_AB_BC[j++] = B.getLength(i);
        }
        if (i == B.getDimension()-1)
        {
            assert(j == ndim_AB+ndim_BC);
        }
    }

    j = 0;
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

    Tensor<T> ar(ndim_AB+ndim_AC, len_AB_AC);
    Tensor<T> br(ndim_AB+ndim_BC, len_AB_BC);
    Tensor<T> cr(ndim_AC+ndim_BC, len_AC_BC);

    Matrix<T> am, bm, cm;
    Scalar<T> alp(alpha);
    Scalar<T> zero;

    Matricize(ar, am, ndim_AB);
    Matricize(br, bm, ndim_AB);
    Matricize(cr, cm, ndim_AC);
    am.setTrans(BLIS_TRANSPOSE);

    Normalize(ar, idx_AB_AC);
    Normalize(br, idx_AB_BC);
    Normalize(cr, idx_AC_BC);

    tensor_transpose_impl<T>(1.0, A, idx_A, 0.0, ar, idx_AB_AC);
    tensor_transpose_impl<T>(1.0, B, idx_B, 0.0, br, idx_AB_BC);
    bli_gemm(alp, am, bm, zero, cm);
    tensor_transpose_impl<T>(1.0, cr, idx_AC_BC, beta, C, idx_C);

    return 0;
}

template
int tensor_contract_blas<   float>(   float alpha, const Tensor<   float>& A, const std::string& idx_A,
                                                   const Tensor<   float>& B, const std::string& idx_B,
                                      float  beta,       Tensor<   float>& C, const std::string& idx_C);

template
int tensor_contract_blas<  double>(  double alpha, const Tensor<  double>& A, const std::string& idx_A,
                                                   const Tensor<  double>& B, const std::string& idx_B,
                                     double  beta,       Tensor<  double>& C, const std::string& idx_C);

template
int tensor_contract_blas<sComplex>(sComplex alpha, const Tensor<sComplex>& A, const std::string& idx_A,
                                                   const Tensor<sComplex>& B, const std::string& idx_B,
                                   sComplex  beta,       Tensor<sComplex>& C, const std::string& idx_C);

template
int tensor_contract_blas<dComplex>(dComplex alpha, const Tensor<dComplex>& A, const std::string& idx_A,
                                                   const Tensor<dComplex>& B, const std::string& idx_B,
                                   dComplex  beta,       Tensor<dComplex>& C, const std::string& idx_C);

}
}
