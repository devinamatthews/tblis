#include "tblis.hpp"
#include "impl/tensor_impl.hpp"

using namespace std;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_dot_reference(const Tensor<T>& A, const std::string& idx_A,
                         const Tensor<T>& B, const std::string& idx_B, T& val)
{
    Iterator<2> iter_AB(A.lengths(), A.strides(), B.strides());

    const T* restrict A_ = A.data();
    const T* restrict B_ = B.data();

    for (val = 0.0;iter_AB.next(A_, B_);)
    {
        assert (A_-A.data() >= 0 && A_-A.data() < A.size());
        assert (B_-B.data() >= 0 && B_-B.data() < B.size());
        val += (*A_)*(*B_);
    }

    return 0;
}

template
int tensor_dot_reference<   float>(const Tensor<   float>& A, const std::string& idx_A,
                                   const Tensor<   float>& B, const std::string& idx_B,    float& val);

template
int tensor_dot_reference<  double>(const Tensor<  double>& A, const std::string& idx_A,
                                   const Tensor<  double>& B, const std::string& idx_B,   double& val);

template
int tensor_dot_reference<sComplex>(const Tensor<sComplex>& A, const std::string& idx_A,
                                   const Tensor<sComplex>& B, const std::string& idx_B, sComplex& val);

template
int tensor_dot_reference<dComplex>(const Tensor<dComplex>& A, const std::string& idx_A,
                                   const Tensor<dComplex>& B, const std::string& idx_B, dComplex& val);

}
}
