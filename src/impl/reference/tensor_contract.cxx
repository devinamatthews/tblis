#include "impl/tensor_impl.hpp"
#include "util/iterator.hpp"

using namespace std;
using namespace blis;
using namespace tensor::util;

namespace tensor
{
namespace impl
{

template <typename T>
int tensor_contract_reference(T alpha, const Tensor<T>& A, const std::string& idx_A,
                                       const Tensor<T>& B, const std::string& idx_B,
                              T  beta,       Tensor<T>& C, const std::string& idx_C)
{
    string idx_AB, idx_AC, idx_BC;

    gint_t ndim_AB  = set_intersection(idx_A, idx_B, idx_AB).size();
    gint_t ndim_AC  = set_intersection(idx_A, idx_C, idx_AC).size();
    gint_t ndim_BC  = set_intersection(idx_B, idx_C, idx_BC).size();

    vector<inc_t> len_AB(ndim_AB);
    vector<inc_t> len_AC(ndim_AC);
    vector<inc_t> len_BC(ndim_BC);

    vector<inc_t> stride_A_AB(ndim_AB);
    vector<inc_t> stride_B_AB(ndim_AB);
    vector<inc_t> stride_A_AC(ndim_AC);
    vector<inc_t> stride_C_AC(ndim_AC);
    vector<inc_t> stride_B_BC(ndim_BC);
    vector<inc_t> stride_C_BC(ndim_BC);

    for (gint_t i = 0, k = 0, l = 0;i < A.getDimension();i++)
    {
        if (k < ndim_AB && idx_A[i] == idx_AB[k])
        {
            len_AB[k] = A.getLength(i);
            stride_A_AB[k++] = A.getStride(i);
        }
        else if (l < ndim_AC && idx_A[i] == idx_AC[l])
        {
            len_AC[l] = A.getLength(i);
            stride_A_AC[l++] = A.getStride(i);
        }
    }

    for (gint_t i = 0, k = 0, l = 0;i < B.getDimension();i++)
    {
        if (k < ndim_AB && idx_B[i] == idx_AB[k])
        {
            stride_B_AB[k++] = B.getStride(i);
        }
        else if (l < ndim_BC && idx_B[i] == idx_BC[l])
        {
            len_BC[l] = B.getLength(i);
            stride_B_BC[l++] = B.getStride(i);
        }
    }

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

    Iterator iter_AB(len_AB, stride_A_AB, stride_B_AB);
    Iterator iter_AC(len_AC, stride_A_AC, stride_C_AC);
    Iterator iter_BC(len_BC, stride_B_BC, stride_C_BC);

    const T* restrict A_ = A.getData();
    const T* restrict B_ = B.getData();
          T* restrict C_ = C.getData();

    while (iter_AC.nextIteration(A_, C_))
    {
        while (iter_BC.nextIteration(B_, C_))
        {
            T temp = T();

            if (alpha != 0.0)
            {
                while (iter_AB.nextIteration(A_, B_))
                {
                    assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
                    assert (B_-B.getData() >= 0 && B_-B.getData() < B.getDataSize());
                    temp += (*A_)*(*B_);
                }
                temp *= alpha;
            }

            if (beta == 0.0)
            {
                assert (C_-C.getData() >= 0 && C_-C.getData() < C.getDataSize());
                *C_ = temp;
            }
            else if (beta == 1.0)
            {
                assert (C_-C.getData() >= 0 && C_-C.getData() < C.getDataSize());
                *C_ += temp;
            }
            else
            {
                assert (C_-C.getData() >= 0 && C_-C.getData() < C.getDataSize());
                *C_ = temp + beta*(*C_);
            }
        }
    }

    return 0;
}

template
int tensor_contract_reference<   float>(   float alpha, const Tensor<   float>& A, const std::string& idx_A,
                                                        const Tensor<   float>& B, const std::string& idx_B,
                                           float  beta,       Tensor<   float>& C, const std::string& idx_C);

template
int tensor_contract_reference<  double>(  double alpha, const Tensor<  double>& A, const std::string& idx_A,
                                                        const Tensor<  double>& B, const std::string& idx_B,
                                          double  beta,       Tensor<  double>& C, const std::string& idx_C);

template
int tensor_contract_reference<sComplex>(sComplex alpha, const Tensor<sComplex>& A, const std::string& idx_A,
                                                        const Tensor<sComplex>& B, const std::string& idx_B,
                                        sComplex  beta,       Tensor<sComplex>& C, const std::string& idx_C);

template
int tensor_contract_reference<dComplex>(dComplex alpha, const Tensor<dComplex>& A, const std::string& idx_A,
                                                        const Tensor<dComplex>& B, const std::string& idx_B,
                                        dComplex  beta,       Tensor<dComplex>& C, const std::string& idx_C);

}
}
