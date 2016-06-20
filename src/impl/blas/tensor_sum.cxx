#include "tblis.hpp"
#include "impl/tensor_impl.hpp"

using namespace std;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_sum_blas(T alpha, const Tensor<T>& A, const std::string& idx_A,
                    T  beta,       Tensor<T>& B, const std::string& idx_B)
{
    string idx_A_not_AB(A.dimension(), 0);
    string idx_B_not_AB(B.dimension(), 0);
    string idx_AB(A.dimension(), 0);

    gint_t ndim_AB =
        set_intersection(idx_A.begin(), idx_A.end(),
                         idx_B.begin(), idx_B.end(),
                         idx_AB.begin()) - idx_AB.begin();

    idx_AB.resize(ndim_AB);
    vector<dim_t> len_AB(ndim_AB);

    gint_t j = 0;
    for (gint_t i = 0;i < A.dimension();i++)
    {
        if (ndim_AB > j && idx_A[i] == idx_AB[j])
        {
            len_AB[j++] = A.length(i);
        }
        if (i == A.dimension()-1)
        {
            assert(j == ndim_AB);
        }
    }

    Tensor<T> c(ndim_AB, len_AB);
    tensor_trace_impl<T>(1.0, A, idx_A, 0.0, c, idx_AB);
    tensor_replicate_impl<T>(alpha, c, idx_AB, beta, B, idx_B);

    return 0;
}

template
int tensor_sum_blas<   float>(   float alpha, const Tensor<   float>& A, const std::string& idx_A,
                                 float  beta,       Tensor<   float>& B, const std::string& idx_B);

template
int tensor_sum_blas<  double>(  double alpha, const Tensor<  double>& A, const std::string& idx_A,
                                double  beta,       Tensor<  double>& B, const std::string& idx_B);

template
int tensor_sum_blas<scomplex>(scomplex alpha, const Tensor<scomplex>& A, const std::string& idx_A,
                              scomplex  beta,       Tensor<scomplex>& B, const std::string& idx_B);

template
int tensor_sum_blas<dcomplex>(dcomplex alpha, const Tensor<dcomplex>& A, const std::string& idx_A,
                              dcomplex  beta,       Tensor<dcomplex>& B, const std::string& idx_B);

}
}
