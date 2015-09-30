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
int tensor_scale_reference(T alpha, Tensor<T>& A, const std::string& idx_A)
{
    Iterator iter_A(A.getLengths(), A.getStrides());

    T* restrict A_ = A.getData();

    if (alpha == 0.0)
    {
        while (iter_A.nextIteration(A_))
        {
            assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
            *A_ = 0.0;
        }
    }
    else if (alpha == 1.0)
    {
        // do nothing
    }
    else
    {
        while (iter_A.nextIteration(A_))
        {
            assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
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
