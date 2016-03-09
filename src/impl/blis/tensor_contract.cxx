#include "tblis.hpp"
#include "impl/tensor_impl.hpp"
#include "util/util.hpp"

using namespace std;
using namespace tblis::util;
using namespace tblis::blis_like;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_contract_blis(T alpha, const Tensor<T>& A, const std::string& idx_A,
                                  const Tensor<T>& B, const std::string& idx_B,
                         T  beta,       Tensor<T>& C, const std::string& idx_C)
{
    gint_t dim_A = A.getDimension();
    gint_t dim_B = B.getDimension();
    gint_t dim_C = C.getDimension();

    gint_t dim_M = (dim_A+dim_C-dim_B)/2;
    gint_t dim_N = (dim_B+dim_C-dim_A)/2;
    gint_t dim_K = (dim_A+dim_B-dim_C)/2;

    vector<dim_t> len_M(dim_M);
    vector<dim_t> len_N(dim_N);
    vector<dim_t> len_K(dim_K);

    vector<inc_t> stride_M_A(dim_M);
    vector<inc_t> stride_M_C(dim_M);
    vector<inc_t> stride_N_B(dim_N);
    vector<inc_t> stride_N_C(dim_N);
    vector<inc_t> stride_K_A(dim_K);
    vector<inc_t> stride_K_B(dim_K);

    dim_t m = 1;
    dim_t n = 1;
    dim_t k = 1;

    gint_t i_M = 0;
    gint_t i_N = 0;
    gint_t i_K = 0;

    for (gint_t i = 0;i < dim_A;i++)
    {
        for (gint_t j = 0;j < dim_B;j++)
        {
            if (idx_A[i] == idx_B[j])
            {
                stride_K_A[i_K] = A.getStride(i);
                stride_K_B[i_K] = B.getStride(j);
                k *= (len_K[i_K] = A.getLength(i));
                i_K++;
            }
        }
    }

    for (gint_t i = 0;i < dim_A;i++)
    {
        for (gint_t j = 0;j < dim_C;j++)
        {
            if (idx_A[i] == idx_C[j])
            {
                stride_M_A[i_M] = A.getStride(i);
                stride_M_C[i_M] = C.getStride(j);
                m *= (len_M[i_M] = A.getLength(i));
                i_M++;
            }
        }
    }

    for (gint_t i = 0;i < dim_B;i++)
    {
        for (gint_t j = 0;j < dim_C;j++)
        {
            if (idx_B[i] == idx_C[j])
            {
                stride_N_B[i_N] = B.getStride(i);
                stride_N_C[i_N] = C.getStride(j);
                n *= (len_N[i_N] = B.getLength(i));
                i_N++;
            }
        }
    }

    vector<inc_t> scat_M_A(m);
    vector<inc_t> scat_M_C(m);
    vector<inc_t> scat_N_B(n);
    vector<inc_t> scat_N_C(n);
    vector<inc_t> scat_K_A(k);
    vector<inc_t> scat_K_B(k);

    Iterator it_M_A(len_M, stride_M_A);
    inc_t idx_M_A = 0;
    for (dim_t i = 0;it_M_A.nextIteration(idx_M_A);i++) scat_M_A[i] = idx_M_A;

    Iterator it_M_C(len_M, stride_M_C);
    inc_t idx_M_C = 0;
    for (dim_t i = 0;it_M_C.nextIteration(idx_M_C);i++) scat_M_C[i] = idx_M_C;

    Iterator it_N_B(len_N, stride_N_B);
    inc_t idx_N_B = 0;
    for (dim_t i = 0;it_N_B.nextIteration(idx_N_B);i++) scat_N_B[i] = idx_N_B;

    Iterator it_N_C(len_N, stride_N_C);
    inc_t idx_N_C = 0;
    for (dim_t i = 0;it_N_C.nextIteration(idx_N_C);i++) scat_N_C[i] = idx_N_C;

    Iterator it_K_A(len_K, stride_K_A);
    inc_t idx_K_A = 0;
    for (dim_t i = 0;it_K_A.nextIteration(idx_K_A);i++) scat_K_A[i] = idx_K_A;

    Iterator it_K_B(len_K, stride_K_B);
    inc_t idx_K_B = 0;
    for (dim_t i = 0;it_K_B.nextIteration(idx_K_B);i++) scat_K_B[i] = idx_K_B;

    ScatterMatrix<T> as(m, k, const_cast<T*>(A.getData()), scat_M_A.data(), scat_K_A.data());
    ScatterMatrix<T> bs(k, n, const_cast<T*>(B.getData()), scat_K_B.data(), scat_N_B.data());
    ScatterMatrix<T> cs(m, n,                C.getData() , scat_M_C.data(), scat_N_C.data());

    tblis_gemm(alpha, as, bs, beta, cs);

    return 0;
}

template
int tensor_contract_blis<   float>(   float alpha, const Tensor<   float>& A, const std::string& idx_A,
                                                   const Tensor<   float>& B, const std::string& idx_B,
                                      float  beta,       Tensor<   float>& C, const std::string& idx_C);

template
int tensor_contract_blis<  double>(  double alpha, const Tensor<  double>& A, const std::string& idx_A,
                                                   const Tensor<  double>& B, const std::string& idx_B,
                                     double  beta,       Tensor<  double>& C, const std::string& idx_C);

template
int tensor_contract_blis<sComplex>(sComplex alpha, const Tensor<sComplex>& A, const std::string& idx_A,
                                                   const Tensor<sComplex>& B, const std::string& idx_B,
                                   sComplex  beta,       Tensor<sComplex>& C, const std::string& idx_C);

template
int tensor_contract_blis<dComplex>(dComplex alpha, const Tensor<dComplex>& A, const std::string& idx_A,
                                                   const Tensor<dComplex>& B, const std::string& idx_B,
                                   dComplex  beta,       Tensor<dComplex>& C, const std::string& idx_C);

}
}
