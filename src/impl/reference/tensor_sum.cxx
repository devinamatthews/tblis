#include "tblis.hpp"
#include "impl/tensor_impl.hpp"

using namespace std;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_sum_reference(T alpha, const Tensor<T>& A, const std::string& idx_A,
                         T  beta,       Tensor<T>& B, const std::string& idx_B)
{
    string idx_A_only, idx_B_only;
    string idx_AB;

    gint_t ndim_AB = util::set_intersection(idx_A, idx_B, idx_AB).size();

    gint_t ndim_A  = util::set_difference(idx_A, idx_AB, idx_A_only).size();
    gint_t ndim_B  = util::set_difference(idx_B, idx_AB, idx_B_only).size();

    vector<inc_t> len_A(ndim_A);
    vector<inc_t> len_B(ndim_B);
    vector<inc_t> len_AB(ndim_AB);

    vector<inc_t> stride_A_A(ndim_A);
    vector<inc_t> stride_B_B(ndim_B);
    vector<inc_t> stride_A_AB(ndim_AB);
    vector<inc_t> stride_B_AB(ndim_AB);

    for (gint_t i = 0, j = 0, k = 0;i < A.getDimension();i++)
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
    }

    for (gint_t i = 0, j = 0, k = 0;i < B.getDimension();i++)
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
    }

    Iterator iter_A(len_A, stride_A_A);
    Iterator iter_B(len_B, stride_B_B);
    Iterator iter_AB(len_AB, stride_A_AB, stride_B_AB);

    const T* restrict A_ = A.getData();
          T* restrict B_ = B.getData();

    while (iter_AB.nextIteration(A_, B_))
    {
        T temp = T();

        if (alpha != 0.0)
        {
            while (iter_A.nextIteration(A_))
            {
                assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
                temp += *A_;
            }
            temp *= alpha;
        }

        if (beta == 0.0)
        {
            while (iter_B.nextIteration(B_))
            {
                assert (B_-B.getData() >= 0 && B_-B.getData() < B.getDataSize());
                *B_ = temp;
            }
        }
        else if (beta == 1.0)
        {
            while (iter_B.nextIteration(B_))
            {
                assert (B_-B.getData() >= 0 && B_-B.getData() < B.getDataSize());
                *B_ += temp;
            }
        }
        else
        {
            while (iter_B.nextIteration(B_))
            {
                assert (B_-B.getData() >= 0 && B_-B.getData() < B.getDataSize());
                *B_ = temp + beta*(*B_);
            }
        }
    }

    return 0;
}

template
int tensor_sum_reference<   float>(   float alpha, const Tensor<   float>& A, const std::string& idx_A,
                                      float  beta,       Tensor<   float>& B, const std::string& idx_B);

template
int tensor_sum_reference<  double>(  double alpha, const Tensor<  double>& A, const std::string& idx_A,
                                     double  beta,       Tensor<  double>& B, const std::string& idx_B);

template
int tensor_sum_reference<sComplex>(sComplex alpha, const Tensor<sComplex>& A, const std::string& idx_A,
                                   sComplex  beta,       Tensor<sComplex>& B, const std::string& idx_B);

template
int tensor_sum_reference<dComplex>(dComplex alpha, const Tensor<dComplex>& A, const std::string& idx_A,
                                   dComplex  beta,       Tensor<dComplex>& B, const std::string& idx_B);

}
}
