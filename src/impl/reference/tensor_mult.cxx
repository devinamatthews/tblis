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
int tensor_mult_reference(T alpha, const Tensor<T>& A, const std::string& idx_A,
                                   const Tensor<T>& B, const std::string& idx_B,
                          T  beta,       Tensor<T>& C, const std::string& idx_C)
{
    string idx_A_only, idx_B_only, idx_C_only;
    string idx_AB, idx_AC, idx_BC;
    string idx_ABC;

    set_intersection(idx_A, idx_B, idx_AB);
    set_intersection(idx_A, idx_C, idx_AC);
    set_intersection(idx_B, idx_C, idx_BC);

    gint_t ndim_ABC = set_intersection(idx_AB, idx_BC, idx_ABC).size();

    gint_t ndim_A   = set_difference(
                      set_difference(idx_A, idx_AB, idx_A_only), idx_AC).size();
    gint_t ndim_B   = set_difference(
                      set_difference(idx_B, idx_AB, idx_B_only), idx_BC).size();
    gint_t ndim_C   = set_difference(
                      set_difference(idx_C, idx_AC, idx_C_only), idx_BC).size();

    gint_t ndim_AB  = set_difference(idx_AB, idx_ABC).size();
    gint_t ndim_AC  = set_difference(idx_AC, idx_ABC).size();
    gint_t ndim_BC  = set_difference(idx_BC, idx_ABC).size();

    vector<inc_t> len_A(ndim_A);
    vector<inc_t> len_B(ndim_B);
    vector<inc_t> len_C(ndim_C);
    vector<inc_t> len_AB(ndim_AB);
    vector<inc_t> len_AC(ndim_AC);
    vector<inc_t> len_BC(ndim_BC);
    vector<inc_t> len_ABC(ndim_ABC);

    vector<inc_t> stride_A_A(ndim_A);
    vector<inc_t> stride_B_B(ndim_B);
    vector<inc_t> stride_C_C(ndim_C);
    vector<inc_t> stride_A_AB(ndim_AB);
    vector<inc_t> stride_B_AB(ndim_AB);
    vector<inc_t> stride_A_AC(ndim_AC);
    vector<inc_t> stride_C_AC(ndim_AC);
    vector<inc_t> stride_B_BC(ndim_BC);
    vector<inc_t> stride_C_BC(ndim_BC);
    vector<inc_t> stride_A_ABC(ndim_ABC);
    vector<inc_t> stride_B_ABC(ndim_ABC);
    vector<inc_t> stride_C_ABC(ndim_ABC);

    for (gint_t i = 0, j = 0, k = 0, l = 0, m = 0;i < A.getDimension();i++)
    {
        if (j < ndim_A && idx_A[i] == idx_A_only[j])
        {
            len_A[j] = A.getLength(i);
            stride_A_A[j++] = A.getStride(i);
        }
        else if (k < ndim_AB && idx_A[i] == idx_AB[k])
        {
            len_AB[k] = A.getLength(i);
            stride_A_AB[k++] = A.getStride(i);
        }
        else if (l < ndim_AC && idx_A[i] == idx_AC[l])
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

    for (gint_t i = 0, j = 0, k = 0, l = 0, m = 0;i < B.getDimension();i++)
    {
        if (j < ndim_B && idx_B[i] == idx_B_only[j])
        {
            len_B[j] = B.getLength(i);
            stride_B_B[j++] = B.getStride(i);
        }
        else if (k < ndim_AB && idx_B[i] == idx_AB[k])
        {
            stride_B_AB[k++] = B.getStride(i);
        }
        else if (l < ndim_BC && idx_B[i] == idx_BC[l])
        {
            len_BC[l] = B.getLength(i);
            stride_B_BC[l++] = B.getStride(i);
        }
        else if (m < ndim_ABC && idx_B[i] == idx_ABC[m])
        {
            stride_B_ABC[m++] = B.getStride(i);
        }
    }

    for (gint_t i = 0, j = 0, k = 0, l = 0, m = 0;i < C.getDimension();i++)
    {
        if (j < ndim_C && idx_C[i] == idx_C_only[j])
        {
            len_C[j] = C.getLength(i);
            stride_C_C[j++] = C.getStride(i);
        }
        else if (k < ndim_AC && idx_C[i] == idx_AC[k])
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

    Iterator iter_A(len_A, stride_A_A);
    Iterator iter_B(len_B, stride_B_B);
    Iterator iter_C(len_C, stride_C_C);
    Iterator iter_AB(len_AB, stride_A_AB, stride_B_AB);
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
            while (iter_BC.nextIteration(B_, C_))
            {
                T temp = T();

                if (alpha != 0.0)
                {
                    while (iter_AB.nextIteration(A_, B_))
                    {
                        T temp_A = T();
                        while (iter_A.nextIteration(A_))
                        {
                            assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
                            temp_A += *A_;
                        }

                        T temp_B = T();
                        while (iter_B.nextIteration(B_))
                        {
                            assert (B_-B.getData() >= 0 && B_-B.getData() < B.getDataSize());
                            temp_B += *B_;
                        }

                        temp += temp_A*temp_B;
                    }

                    temp *= alpha;
                }

                if (beta == 0.0)
                {
                    while (iter_C.nextIteration(C_))
                    {
                        assert (C_-C.getData() >= 0 && C_-C.getData() < C.getDataSize());
                        *C_ = temp;
                    }
                }
                else if (beta == 1.0)
                {
                    while (iter_C.nextIteration(C_))
                    {
                        assert (C_-C.getData() >= 0 && C_-C.getData() < C.getDataSize());
                        *C_ += temp;
                    }
                }
                else
                {
                    while (iter_C.nextIteration(C_))
                    {
                        assert (C_-C.getData() >= 0 && C_-C.getData() < C.getDataSize());
                        *C_ = temp + beta*(*C_);
                    }
                }
            }
        }
    }

    return 0;
}

template
int tensor_mult_reference<   float>(   float alpha, const Tensor<   float>& A, const std::string& idx_A,
                                                    const Tensor<   float>& B, const std::string& idx_B,
                                       float  beta,       Tensor<   float>& C, const std::string& idx_C);

template
int tensor_mult_reference<  double>(  double alpha, const Tensor<  double>& A, const std::string& idx_A,
                                                    const Tensor<  double>& B, const std::string& idx_B,
                                      double  beta,       Tensor<  double>& C, const std::string& idx_C);

template
int tensor_mult_reference<sComplex>(sComplex alpha, const Tensor<sComplex>& A, const std::string& idx_A,
                                                    const Tensor<sComplex>& B, const std::string& idx_B,
                                    sComplex  beta,       Tensor<sComplex>& C, const std::string& idx_C);

template
int tensor_mult_reference<dComplex>(dComplex alpha, const Tensor<dComplex>& A, const std::string& idx_A,
                                                    const Tensor<dComplex>& B, const std::string& idx_B,
                                    dComplex  beta,       Tensor<dComplex>& C, const std::string& idx_C);

}
}
