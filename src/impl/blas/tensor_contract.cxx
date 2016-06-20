#include "tblis.hpp"
#include "impl/tensor_impl.hpp"

#include "external/lawrap/blas.h"

using namespace std;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_contract_blas(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                  const const_tensor_view<T>& B, const std::string& idx_B,
                         T  beta,             tensor_view<T>& C, const std::string& idx_C)
{
    string idx_AB_AC(A.dimension(), 0);
    string idx_AB_BC(B.dimension(), 0);
    string idx_AC_BC(C.dimension(), 0);

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

    assert(ndim_AB+ndim_AC == A.dimension());
    assert(ndim_AB+ndim_BC == B.dimension());
    assert(ndim_AC+ndim_BC == C.dimension());

    vector<dim_t> len_AB_AC(ndim_AB+ndim_AC);
    vector<dim_t> len_AB_BC(ndim_AB+ndim_BC);
    vector<dim_t> len_AC_BC(ndim_AC+ndim_BC);

    gint_t j = 0;
    for (gint_t i = 0;i < A.dimension();i++)
    {
        if (ndim_AB+ndim_AC > j && idx_A[i] == idx_AB_AC[j])
        {
            len_AB_AC[j++] = A.length(i);
        }
    }
    for (gint_t i = 0;i < A.dimension();i++)
    {
        if (ndim_AB+ndim_AC > j && idx_A[i] == idx_AB_AC[j])
        {
            len_AB_AC[j++] = A.length(i);
        }
        if (i == A.dimension()-1)
        {
            assert(j == ndim_AB+ndim_AC);
        }
    }

    j = 0;
    for (gint_t i = 0;i < B.dimension();i++)
    {
        if (ndim_AB+ndim_BC > j && idx_B[i] == idx_AB_BC[j])
        {
            len_AB_BC[j++] = B.length(i);
        }
    }
    for (gint_t i = 0;i < B.dimension();i++)
    {
        if (ndim_AB+ndim_BC > j && idx_B[i] == idx_AB_BC[j])
        {
            len_AB_BC[j++] = B.length(i);
        }
        if (i == B.dimension()-1)
        {
            assert(j == ndim_AB+ndim_BC);
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
    for (gint_t i = 0;i < C.dimension();i++)
    {
        if (ndim_AC+ndim_BC > j && idx_C[i] == idx_AC_BC[j])
        {
            len_AC_BC[j++] = C.length(i);
        }
        if (i == C.dimension()-1)
        {
            assert(j == ndim_AC+ndim_BC);
        }
    }

    tensor<T> ar(len_AB_AC);
    tensor<T> br(len_AB_BC);
    tensor<T> cr(len_AC_BC);

    matrix_view<T> am, bm, cm;

    matricize(ar, am, ndim_AB);
    matricize(br, bm, ndim_AB);
    matricize(cr, cm, ndim_AC);
    am.transpose(true);

    normalize(ar, idx_AB_AC);
    normalize(br, idx_AB_BC);
    normalize(cr, idx_AC_BC);

    tensor_transpose_impl<T>(1.0, A, idx_A, 0.0, ar, idx_AB_AC);
    tensor_transpose_impl<T>(1.0, B, idx_B, 0.0, br, idx_AB_BC);
    LAWrap::gemm('T', 'N', cm.length(), cm.width(), am.length(),
                 alpha, am.data(), 1, am.stride(1),
                        bm.data(), 1, bm.stride(1),
                   0.0, cm.data(), 1, cm.stride(1));
    tensor_transpose_impl<T>(1.0, cr, idx_AC_BC, beta, C, idx_C);

    return 0;
}

template
int tensor_contract_blas<   float>(   float alpha, const const_tensor_view<   float>& A, const std::string& idx_A,
                                                   const const_tensor_view<   float>& B, const std::string& idx_B,
                                      float  beta,             tensor_view<   float>& C, const std::string& idx_C);

template
int tensor_contract_blas<  double>(  double alpha, const const_tensor_view<  double>& A, const std::string& idx_A,
                                                   const const_tensor_view<  double>& B, const std::string& idx_B,
                                     double  beta,             tensor_view<  double>& C, const std::string& idx_C);

template
int tensor_contract_blas<scomplex>(scomplex alpha, const const_tensor_view<scomplex>& A, const std::string& idx_A,
                                                   const const_tensor_view<scomplex>& B, const std::string& idx_B,
                                   scomplex  beta,             tensor_view<scomplex>& C, const std::string& idx_C);

template
int tensor_contract_blas<dcomplex>(dcomplex alpha, const const_tensor_view<dcomplex>& A, const std::string& idx_A,
                                                   const const_tensor_view<dcomplex>& B, const std::string& idx_B,
                                   dcomplex  beta,             tensor_view<dcomplex>& C, const std::string& idx_C);

}
}
