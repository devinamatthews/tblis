#include "tblis.hpp"
#include "impl/tensor_impl.hpp"

using namespace std;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_weight_blas(T alpha, const Tensor<T>& A, const std::string& idx_A,
                                const Tensor<T>& B, const std::string& idx_B,
                       T  beta,       Tensor<T>& C, const std::string& idx_C)
{
    gint_t ndim_max = max(A.dimension(),
                      max(B.dimension(),
                          C.dimension()));

    string idx_A_not_ABC(A.dimension(), 0);
    string idx_B_not_ABC(B.dimension(), 0);
    string idx_C_not_ABC(C.dimension(), 0);
    string idx_AC_BC(C.dimension(), 0);
    string idx_ABC(ndim_max, 0);

    gint_t ndim_ABC =
        set_intersection(idx_A.begin(), idx_A.end(),
                         idx_B.begin(), idx_B.end(),
                         idx_ABC.begin()) - idx_ABC.begin();
    ndim_ABC =
        util::set_intersection(idx_ABC.begin(), idx_ABC.begin()+ndim_ABC,
                               idx_C.begin(), idx_C.end()) - idx_ABC.begin();

    idx_ABC.resize(ndim_ABC);

    gint_t ndim_A_not_ABC =
        set_difference(idx_A.begin()  , idx_A.end(),
                       idx_ABC.begin(), idx_ABC.end(),
                       idx_A_not_ABC.begin()) - idx_A_not_ABC.begin();

    gint_t ndim_B_not_ABC =
        set_difference(idx_B.begin()  , idx_B.end(),
                       idx_ABC.begin(), idx_ABC.end(),
                       idx_B_not_ABC.begin()) - idx_B_not_ABC.begin();

    gint_t ndim_C_not_ABC =
        set_difference(idx_C.begin()  , idx_C.end(),
                       idx_ABC.begin(), idx_ABC.end(),
                       idx_C_not_ABC.begin()) - idx_C_not_ABC.begin();

    idx_A_not_ABC.resize(ndim_A_not_ABC);
    idx_B_not_ABC.resize(ndim_B_not_ABC);
    idx_C_not_ABC.resize(ndim_C_not_ABC);

    gint_t ndim_AC =
        set_intersection(idx_A_not_ABC.begin(), idx_A_not_ABC.end(),
                         idx_C_not_ABC.begin(), idx_C_not_ABC.end(),
                         idx_AC_BC.begin()) - idx_AC_BC.begin();

    gint_t ndim_BC =
        set_intersection(idx_B_not_ABC.begin(), idx_B_not_ABC.end(),
                         idx_C_not_ABC.begin(), idx_C_not_ABC.end(),
                         idx_AC_BC.begin()+ndim_AC) - (idx_AC_BC.begin()+ndim_AC);

    idx_AC_BC.resize(ndim_AC+ndim_BC);

    vector<dim_t> len_A_not_ABC(ndim_A_not_ABC);
    vector<dim_t> len_B_not_ABC(ndim_B_not_ABC);
    vector<dim_t> len_AC_BC(ndim_AC+ndim_BC);

    vector<gint_t> dims_A_ABC(ndim_ABC);
    vector<gint_t> dims_B_ABC(ndim_ABC);
    vector<gint_t> dims_C_ABC(ndim_ABC);

    gint_t j = 0;
    for (gint_t i = 0;i < A.dimension();i++)
    {
        if (ndim_A_not_ABC > j && idx_A[i] == idx_A_not_ABC[j])
        {
            len_A_not_ABC[j++] = A.length(i);
        }
    }
    for (gint_t i = 0, k = 0;i < A.dimension();i++)
    {
        if (ndim_A_not_ABC > j && idx_A[i] == idx_A_not_ABC[j])
        {
            len_A_not_ABC[j++] = A.length(i);
        }
        else if (ndim_ABC > k && idx_A[i] == idx_ABC[k])
        {
            dims_A_ABC[k++] = i;
        }
        if (i == A.dimension()-1)
        {
            assert(j == ndim_A_not_ABC);
            assert(k == ndim_ABC);
        }
    }

    j = 0;
    for (gint_t i = 0;i < B.dimension();i++)
    {
        if (ndim_B_not_ABC > j && idx_B[i] == idx_B_not_ABC[j])
        {
            len_B_not_ABC[j++] = B.length(i);
        }
    }
    for (gint_t i = 0, k = 0;i < B.dimension();i++)
    {
        if (ndim_B_not_ABC > j && idx_B[i] == idx_B_not_ABC[j])
        {
            len_B_not_ABC[j++] = B.length(i);
        }
        else if (ndim_ABC > k && idx_B[i] == idx_ABC[k])
        {
            dims_B_ABC[k++] = i;
        }
        if (i == B.dimension()-1)
        {
            assert(j == ndim_B_not_ABC);
            assert(k == ndim_ABC);
        }
    }

    j = 0;
    for (gint_t i = 0;i < C.dimension();i++)
    {
        if (ndim_AC+ndim_BC > j && idx_C[i] == idx_AC_BC[j])
        {
            len_AC_BC[j++] = C.length(i);
        }
    }
    for (gint_t i = 0, k = 0;i < C.dimension();i++)
    {
        if (ndim_AC+ndim_BC > j && idx_C[i] == idx_AC_BC[j])
        {
            len_AC_BC[j++] = C.length(i);
        }
        else if (ndim_ABC > k && idx_C[i] == idx_ABC[k])
        {
            dims_C_ABC[k++] = i;
        }
        if (i == C.dimension()-1)
        {
            assert(j == ndim_AC+ndim_BC);
            assert(k == ndim_ABC);
        }
    }

    Tensor<T> ar(ndim_A_not_ABC, len_A_not_ABC);
    Tensor<T> br(ndim_B_not_ABC, len_B_not_ABC);
    Tensor<T> cr(ndim_AC+ndim_BC, len_AC_BC);

    Matrix<T> am, bm, cm;
    Scalar<T> alp(alpha);
    Scalar<T> zero;

    Matricize(ar, am, 0);
    Matricize(br, bm, 0);
    Matricize(cr, cm, ndim_AC);
    am.transpose(true);

    Normalize(cr, idx_AC_BC);

    Slicer<T> sA(A, dims_A_ABC);
    Slicer<T> sB(B, dims_B_ABC);
    Slicer<T> sC(C, dims_C_ABC);

    Tensor<T> a, b, c;

    while (sA.nextSlice(a) + sB.nextSlice(b) + sC.nextSlice(c))
    {
        tensor_trace_impl<T>(1.0, a, idx_A_not_ABC, 0.0, ar, idx_A_not_ABC);
        tensor_trace_impl<T>(1.0, b, idx_B_not_ABC, 0.0, br, idx_B_not_ABC);
        bli_setm(zero, cm);
        bli_ger(alp, am, bm, cm);
        tensor_replicate_impl<T>(1.0, cr, idx_AC_BC, beta, c, idx_C_not_ABC);
    }

    return 0;
}

template
int tensor_weight_blas<   float>(   float alpha, const Tensor<   float>& A, const std::string& idx_A,
                                                 const Tensor<   float>& B, const std::string& idx_B,
                                    float  beta,       Tensor<   float>& C, const std::string& idx_C);

template
int tensor_weight_blas<  double>(  double alpha, const Tensor<  double>& A, const std::string& idx_A,
                                                 const Tensor<  double>& B, const std::string& idx_B,
                                   double  beta,       Tensor<  double>& C, const std::string& idx_C);

template
int tensor_weight_blas<sComplex>(sComplex alpha, const Tensor<sComplex>& A, const std::string& idx_A,
                                                 const Tensor<sComplex>& B, const std::string& idx_B,
                                 sComplex  beta,       Tensor<sComplex>& C, const std::string& idx_C);

template
int tensor_weight_blas<dComplex>(dComplex alpha, const Tensor<dComplex>& A, const std::string& idx_A,
                                                 const Tensor<dComplex>& B, const std::string& idx_B,
                                 dComplex  beta,       Tensor<dComplex>& C, const std::string& idx_C);

}
}
