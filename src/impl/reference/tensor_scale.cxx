#include "tblis.hpp"
#include "impl/tensor_impl.hpp"

using namespace std;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_scale_reference(T alpha, Tensor<T>& A, const std::string& idx_A)
{
    Iterator<> iter_A(A.lengths(), A.strides());

    T* restrict A_ = A.data();

    if (alpha == 0.0)
    {
        while (iter_A.next(A_))
        {
            assert (A_-A.data() >= 0 && A_-A.data() < A.size());
            *A_ = 0.0;
        }
    }
    else if (alpha == 1.0)
    {
        // do nothing
    }
    else
    {
        while (iter_A.next(A_))
        {
            assert (A_-A.data() >= 0 && A_-A.data() < A.size());
            *A_ *= alpha;
        }
    }

    return 0;
}

template
int tensor_scale_reference<   float>(   float alpha, Tensor<   float>& A, const std::string& idx_A);

template
int tensor_scale_reference<  double>(  double alpha, Tensor<  double>& A, const std::string& idx_A);

template
int tensor_scale_reference<sComplex>(sComplex alpha, Tensor<sComplex>& A, const std::string& idx_A);

template
int tensor_scale_reference<dComplex>(dComplex alpha, Tensor<dComplex>& A, const std::string& idx_A);

}
}
