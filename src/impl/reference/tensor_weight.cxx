#include "tblis.hpp"
#include "impl/tensor_impl.hpp"

using namespace std;

namespace tblis
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

    gint_t ndim_ABC = util::set_intersection(idx_A, idx_B  , idx_ABC).size();
    gint_t ndim_AC  = util::set_difference  (idx_A, idx_ABC, idx_AC ).size();
    gint_t ndim_BC  = util::set_difference  (idx_B, idx_ABC, idx_BC ).size();

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

    for (gint_t i = 0, l = 0, m = 0;i < A.dimension();i++)
    {
        if (l < ndim_AC && idx_A[i] == idx_AC[l])
        {
            len_AC[l] = A.length(i);
            stride_A_AC[l++] = A.stride(i);
        }
        else if (m < ndim_ABC && idx_A[i] == idx_ABC[m])
        {
            len_ABC[m] = A.length(i);
            stride_A_ABC[m++] = A.stride(i);
        }
    }

    for (gint_t i = 0, l = 0, m = 0;i < B.dimension();i++)
    {
        if (l < ndim_BC && idx_B[i] == idx_BC[l])
        {
            len_BC[l] = B.length(i);
            stride_B_BC[l++] = B.stride(i);
        }
        else if (m < ndim_ABC && idx_B[i] == idx_ABC[m])
        {
            stride_B_ABC[m++] = B.stride(i);
        }
    }

    for (gint_t i = 0, k = 0, l = 0, m = 0;i < C.dimension();i++)
    {
        if (k < ndim_AC && idx_C[i] == idx_AC[k])
        {
            stride_C_AC[k++] = C.stride(i);
        }
        else if (l < ndim_BC && idx_C[i] == idx_BC[l])
        {
            stride_C_BC[l++] = C.stride(i);
        }
        else if (m < ndim_ABC && idx_C[i] == idx_ABC[m])
        {
            stride_C_ABC[m++] = C.stride(i);
        }
    }

    Iterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    Iterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);
    Iterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

    const T* restrict A_ = A.data();
    const T* restrict B_ = B.data();
          T* restrict C_ = C.data();

    while (iter_ABC.next(A_, B_, C_))
    {
        while (iter_AC.next(A_, C_))
        {
            assert (A_-A.data() >= 0 && A_-A.data() < A.size());
            T temp = alpha*(*A_);

            if (beta == 0.0)
            {
                while (iter_BC.next(B_, C_))
                {
                    assert (B_-B.data() >= 0 && B_-B.data() < B.size());
                    assert (C_-C.data() >= 0 && C_-C.data() < C.size());
                    *C_ = temp*(*B_);
                }
            }
            else if (beta == 1.0)
            {
                while (iter_BC.next(B_, C_))
                {
                    assert (B_-B.data() >= 0 && B_-B.data() < B.size());
                    assert (C_-C.data() >= 0 && C_-C.data() < C.size());
                    *C_ += temp*(*B_);
                }
            }
            else
            {
                while (iter_BC.next(B_, C_))
                {
                    assert (B_-B.data() >= 0 && B_-B.data() < B.size());
                    assert (C_-C.data() >= 0 && C_-C.data() < C.size());
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
int tensor_weight_reference<scomplex>(scomplex alpha, const Tensor<scomplex>& A, const std::string& idx_A,
                                                      const Tensor<scomplex>& B, const std::string& idx_B,
                                      scomplex  beta,       Tensor<scomplex>& C, const std::string& idx_C);

template
int tensor_weight_reference<dcomplex>(dcomplex alpha, const Tensor<dcomplex>& A, const std::string& idx_A,
                                                      const Tensor<dcomplex>& B, const std::string& idx_B,
                                      dcomplex  beta,       Tensor<dcomplex>& C, const std::string& idx_C);

}
}
