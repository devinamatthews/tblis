#include "tblis.hpp"
#include "impl/tensor_impl.hpp"

using namespace std;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_outer_prod_reference(T alpha, const Tensor<T>& A, const std::string& idx_A,
                                         const Tensor<T>& B, const std::string& idx_B,
                                T  beta,       Tensor<T>& C, const std::string& idx_C)
{
    const string& idx_AC = idx_A;
    const string& idx_BC = idx_B;

    gint_t ndim_AC = A.getDimension();
    gint_t ndim_BC = B.getDimension();

    const vector<inc_t>& len_AC = A.getLengths();
    const vector<inc_t>& len_BC = B.getLengths();

    const vector<inc_t>& stride_A_AC = A.getStrides();
    const vector<inc_t>& stride_B_BC = B.getStrides();

    vector<inc_t> stride_C_AC(ndim_AC);
    vector<inc_t> stride_C_BC(ndim_BC);

    for (gint_t i = 0, k = 0, l = 0;i < C.getDimension();i++)
    {
        if (k < ndim_AC && idx_C[i] == idx_AC[k])
        {
            stride_C_AC[k++] = C.getStride(i);
        }
        else if (l < ndim_BC && idx_C[i] == idx_BC[l])
        {
            stride_C_BC[l++] = C.getStride(i);
        }
    }

    Iterator iter_AC(len_AC, stride_A_AC, stride_C_AC);
    Iterator iter_BC(len_BC, stride_B_BC, stride_C_BC);

    const T* restrict A_ = A.getData();
    const T* restrict B_ = B.getData();
          T* restrict C_ = C.getData();

    while (iter_AC.nextIteration(A_, C_))
    {
        assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
        T temp = alpha*(*A_);

        if (beta == 0.0)
        {
            while (iter_BC.nextIteration(B_, C_))
            {
                assert (B_-B.getData() >= 0 && B_-B.getData() < B.getDataSize());
                assert (C_-C.getData() >= 0 && C_-C.getData() < C.getDataSize());
                *C_ = temp*(*B_);
            }
        }
        else if (beta == 1.0)
        {
            while (iter_BC.nextIteration(B_, C_))
            {
                assert (B_-B.getData() >= 0 && B_-B.getData() < B.getDataSize());
                assert (C_-C.getData() >= 0 && C_-C.getData() < C.getDataSize());
                *C_ += temp*(*B_);
            }
        }
        else
        {
            while (iter_BC.nextIteration(B_, C_))
            {
                assert (B_-B.getData() >= 0 && B_-B.getData() < B.getDataSize());
                assert (C_-C.getData() >= 0 && C_-C.getData() < C.getDataSize());
                *C_ = temp*(*B_) + beta*(*C_);
            }
        }
    }

    return 0;
}

template
int tensor_outer_prod_reference<   float>(   float alpha, const Tensor<   float>& A, const std::string& idx_A,
                                                          const Tensor<   float>& B, const std::string& idx_B,
                                             float  beta,       Tensor<   float>& C, const std::string& idx_C);

template
int tensor_outer_prod_reference<  double>(  double alpha, const Tensor<  double>& A, const std::string& idx_A,
                                                          const Tensor<  double>& B, const std::string& idx_B,
                                            double  beta,       Tensor<  double>& C, const std::string& idx_C);

template
int tensor_outer_prod_reference<sComplex>(sComplex alpha, const Tensor<sComplex>& A, const std::string& idx_A,
                                                          const Tensor<sComplex>& B, const std::string& idx_B,
                                          sComplex  beta,       Tensor<sComplex>& C, const std::string& idx_C);

template
int tensor_outer_prod_reference<dComplex>(dComplex alpha, const Tensor<dComplex>& A, const std::string& idx_A,
                                                          const Tensor<dComplex>& B, const std::string& idx_B,
                                          dComplex  beta,       Tensor<dComplex>& C, const std::string& idx_C);

}
}
