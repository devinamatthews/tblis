#include "tblis.hpp"
#include "impl/tensor_impl.hpp"

using namespace std;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_replicate_reference(T alpha, const Tensor<T>& A, const std::string& idx_A,
                               T  beta,       Tensor<T>& B, const std::string& idx_B)
{
    const string& idx_AB = idx_A;
    string idx_B_only;

    gint_t ndim_AB = idx_AB.size();
    gint_t ndim_B  = util::set_difference(idx_B, idx_AB, idx_B_only).size();

    vector<inc_t> len_B(ndim_B);
    const vector<inc_t>& len_AB = A.getLengths();

    vector<inc_t> stride_B_B(ndim_B);
    const vector<inc_t>& stride_A_AB = A.getStrides();
    vector<inc_t> stride_B_AB(ndim_AB);

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

    Iterator iter_B(len_B, stride_B_B);
    Iterator iter_AB(len_AB, stride_A_AB, stride_B_AB);

    const T* restrict A_ = A.getData();
          T* restrict B_ = B.getData();

    while (iter_AB.nextIteration(A_, B_))
    {
        assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
        T temp = alpha*(*A_);

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
int tensor_replicate_reference<   float>(   float alpha, const Tensor<   float>& A, const std::string& idx_A,
                                            float  beta,       Tensor<   float>& B, const std::string& idx_B);

template
int tensor_replicate_reference<  double>(  double alpha, const Tensor<  double>& A, const std::string& idx_A,
                                           double  beta,       Tensor<  double>& B, const std::string& idx_B);

template
int tensor_replicate_reference<sComplex>(sComplex alpha, const Tensor<sComplex>& A, const std::string& idx_A,
                                         sComplex  beta,       Tensor<sComplex>& B, const std::string& idx_B);

template
int tensor_replicate_reference<dComplex>(dComplex alpha, const Tensor<dComplex>& A, const std::string& idx_A,
                                         dComplex  beta,       Tensor<dComplex>& B, const std::string& idx_B);

}
}
