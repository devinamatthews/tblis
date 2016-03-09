#include "tblis.hpp"
#include "impl/tensor_impl.hpp"

using namespace std;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_transpose_reference(T alpha, const Tensor<T>& A, const std::string& idx_A,
                               T  beta,       Tensor<T>& B, const std::string& idx_B)
{
    Iterator iter_AB(A.getLengths(), A.getStrides(), B.getStrides());

    const T* restrict A_ = A.getData();
          T* restrict B_ = B.getData();

    if (alpha == 0.0)
    {
        if (beta == 0.0)
        {
            while (iter_AB.nextIteration(A_, B_))
            {
                assert (B_-B.getData() >= 0 && B_-B.getData() < B.getDataSize());
                *B_ = 0.0;
            }
        }
        else if (beta == 1.0)
        {
            // do nothing
        }
        else
        {
            while (iter_AB.nextIteration(A_, B_))
            {
                assert (B_-B.getData() >= 0 && B_-B.getData() < B.getDataSize());
                *B_ *= beta;
            }
        }
    }
    else if (alpha == 1.0)
    {
        if (beta == 0.0)
        {
            while (iter_AB.nextIteration(A_, B_))
            {
                assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
                assert (B_-B.getData() >= 0 && B_-B.getData() < B.getDataSize());
                *B_ = *A_;
            }
        }
        else if (beta == 1.0)
        {
            while (iter_AB.nextIteration(A_, B_))
            {
                assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
                assert (B_-B.getData() >= 0 && B_-B.getData() < B.getDataSize());
                *B_ += *A_;
            }
        }
        else
        {
            while (iter_AB.nextIteration(A_, B_))
            {
                assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
                assert (B_-B.getData() >= 0 && B_-B.getData() < B.getDataSize());
                *B_ = *A_ + beta*(*B_);
            }
        }
    }
    else
    {
        if (beta == 0.0)
        {
            while (iter_AB.nextIteration(A_, B_))
            {
                assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
                assert (B_-B.getData() >= 0 && B_-B.getData() < B.getDataSize());
                *B_ = alpha*(*A_);
            }
        }
        else if (beta == 1.0)
        {
            while (iter_AB.nextIteration(A_, B_))
            {
                assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
                assert (B_-B.getData() >= 0 && B_-B.getData() < B.getDataSize());
                *B_ += alpha*(*A_);
            }
        }
        else
        {
            while (iter_AB.nextIteration(A_, B_))
            {
                assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
                assert (B_-B.getData() >= 0 && B_-B.getData() < B.getDataSize());
                *B_ = alpha*(*A_) + beta*(*B_);
            }
        }
    }

    return 0;
}

template
int tensor_transpose_reference<   float>(   float alpha, const Tensor<   float>& A, const std::string& idx_A,
                                            float  beta,       Tensor<   float>& B, const std::string& idx_B);

template
int tensor_transpose_reference<  double>(  double alpha, const Tensor<  double>& A, const std::string& idx_A,
                                           double  beta,       Tensor<  double>& B, const std::string& idx_B);

template
int tensor_transpose_reference<sComplex>(sComplex alpha, const Tensor<sComplex>& A, const std::string& idx_A,
                                         sComplex  beta,       Tensor<sComplex>& B, const std::string& idx_B);

template
int tensor_transpose_reference<dComplex>(dComplex alpha, const Tensor<dComplex>& A, const std::string& idx_A,
                                         dComplex  beta,       Tensor<dComplex>& B, const std::string& idx_B);

}
}
