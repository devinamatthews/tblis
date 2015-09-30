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
int tensor_weight_reference(T alpha, const Tensor<T>& A, const std::string& idx_A,
                                     const Tensor<T>& B, const std::string& idx_B,
                            T  beta,       Tensor<T>& C, const std::string& idx_C)
{
    string idx_AC, idx_BC;
    string idx_ABC;

    gint_t ndim_ABC = set_intersection(idx_A, idx_B  , idx_ABC).size();
    gint_t ndim_AC  = set_difference  (idx_A, idx_ABC, idx_AC ).size();
    gint_t ndim_BC  = set_difference  (idx_B, idx_ABC, idx_BC ).size();

    vector<inc_t> len_AC(ndim_AC);
    vector<inc_t> len_BC(ndim_BC);
    vector<inc_t> len_ABC(ndim_ABC);

    vector<inc_t> stride_A_AC(ndim_AC);
    vector<inc_t> stride_C_AC(ndim_AC);
    vector<inc_t> stride_B_BC(ndim_BC);
    vector<inc_t> stride_C_BC(ndim_BC);
    vector<inc_t> stride_A_ABC(ndim_ABC);
    vector<inc_t> stride_B_ABC(ndim_ABC);
    vector<inc_t> stride_C_ABC(ndim_ABC);

    for (gint_t i = 0, l = 0, m = 0;i < A.getDimension();i++)
    {
        if (l < ndim_AC && idx_A[i] == idx_AC[l])
        {
            len_AC[l] = A.getLength(i);
            stride_A_AC[l++] = A.getStride(i);
        }
        else if (m < ndim_ABC && idx_A[i] == idx_ABC[m])
        {
            len_ABC[m] = A.getLength(i);
            stride_A_ABC[m++] = A.getStride(i);
        }
    }

    for (gint_t i = 0, l = 0, m = 0;i < B.getDimension();i++)
    {
        if (l < ndim_BC && idx_B[i] == idx_BC[l])
        {
            len_BC[l] = B.getLength(i);
            stride_B_BC[l++] = B.getStride(i);
        }
        else if (m < ndim_ABC && idx_B[i] == idx_ABC[m])
        {
            stride_B_ABC[m++] = B.getStride(i);
        }
    }

    for (gint_t i = 0, k = 0, l = 0, m = 0;i < C.getDimension();i++)
    {
        if (k < ndim_AC && idx_C[i] == idx_AC[k])
        {
            stride_C_AC[k++] = C.getStride(i);
        }
        else if (l < ndim_BC && idx_C[i] == idx_BC[l])
        {
            stride_C_BC[l++] = C.getStride(i);
        }
        else if (m < ndim_ABC && idx_C[i] == idx_ABC[m])
        {
            stride_C_ABC[m++] = C.getStride(i);
        }
    }

    Iterator iter_AC(len_AC, stride_A_AC, stride_C_AC);
    Iterator iter_BC(len_BC, stride_B_BC, stride_C_BC);
    Iterator iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

    const T* restrict A_ = A.getData();
    const T* restrict B_ = B.getData();
          T* restrict C_ = C.getData();

    while (iter_ABC.nextIteration(A_, B_, C_))
    {
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
    }

    return 0;
}

template
int tensor_weight_reference<   float>(   float alpha, const Tensor<   float>& A, const std::string& idx_A,
                                                      const Tensor<   float>& B, const std::string& idx_B,
                                         float  beta,       Tensor<   float>& C, const std::string& idx_C);

template
int tensor_weight_reference<  double>(  double alpha, const Tensor<  double>& A, const std::string& idx_A,
                                                      const Tensor<  double>& B, const std::string& idx_B,
                                        double  beta,       Tensor<  double>& C, const std::string& idx_C);

template
int tensor_weight_reference<sComplex>(sComplex alpha, const Tensor<sComplex>& A, const std::string& idx_A,
                                                      const Tensor<sComplex>& B, const std::string& idx_B,
                                      sComplex  beta,       Tensor<sComplex>& C, const std::string& idx_C);

template
int tensor_weight_reference<dComplex>(dComplex alpha, const Tensor<dComplex>& A, const std::string& idx_A,
                                                      const Tensor<dComplex>& B, const std::string& idx_B,
                                      dComplex  beta,       Tensor<dComplex>& C, const std::string& idx_C);

}
}
