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

    for (gint_t i = 0, j = 0, k = 0;i < A.dimension();i++)
    {
        if (j < ndim_A && idx_A[i] == idx_A_only[j])
        {
            len_A[j] = A.length(i);
            stride_A_A[j++] = A.stride(i);
        }
        else if (k < ndim_AB && idx_A[i] == idx_AB[k])
        {
            len_AB[k] = A.length(i);
            stride_A_AB[k++] = A.stride(i);
        }
    }

    for (gint_t i = 0, j = 0, k = 0;i < B.dimension();i++)
    {
        if (j < ndim_B && idx_B[i] == idx_B_only[j])
        {
            len_B[j] = B.length(i);
            stride_B_B[j++] = B.stride(i);
        }
        else if (k < ndim_AB && idx_B[i] == idx_AB[k])
        {
            stride_B_AB[k++] = B.stride(i);
        }
    }

    Iterator<1> iter_A(len_A, stride_A_A);
    Iterator<1> iter_B(len_B, stride_B_B);
    Iterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);

    const T* restrict A_ = A.data();
          T* restrict B_ = B.data();

    while (iter_AB.next(A_, B_))
    {
        T temp = T();

        if (alpha != 0.0)
        {
            while (iter_A.next(A_))
            {
                assert (A_-A.data() >= 0 && A_-A.data() < A.size());
                temp += *A_;
            }
            temp *= alpha;
        }

        if (beta == 0.0)
        {
            while (iter_B.next(B_))
            {
                assert (B_-B.data() >= 0 && B_-B.data() < B.size());
                *B_ = temp;
            }
        }
        else if (beta == 1.0)
        {
            while (iter_B.next(B_))
            {
                assert (B_-B.data() >= 0 && B_-B.data() < B.size());
                *B_ += temp;
            }
        }
        else
        {
            while (iter_B.next(B_))
            {
                assert (B_-B.data() >= 0 && B_-B.data() < B.size());
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
