#include "tblis.hpp"
#include "impl/tensor_impl.hpp"

using namespace std;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_contract_reference(T alpha, const Tensor<T>& A, const std::string& idx_A,
                                       const Tensor<T>& B, const std::string& idx_B,
                              T  beta,       Tensor<T>& C, const std::string& idx_C)
{
    string idx_AB, idx_AC, idx_BC;

    gint_t ndim_AB  = util::set_intersection(idx_A, idx_B, idx_AB).size();
    gint_t ndim_AC  = util::set_intersection(idx_A, idx_C, idx_AC).size();
    gint_t ndim_BC  = util::set_intersection(idx_B, idx_C, idx_BC).size();

    vector<inc_t> len_AB(ndim_AB);
    vector<inc_t> len_AC(ndim_AC);
    vector<inc_t> len_BC(ndim_BC);

    vector<inc_t> stride_A_AB(ndim_AB);
    vector<inc_t> stride_B_AB(ndim_AB);
    vector<inc_t> stride_A_AC(ndim_AC);
    vector<inc_t> stride_C_AC(ndim_AC);
    vector<inc_t> stride_B_BC(ndim_BC);
    vector<inc_t> stride_C_BC(ndim_BC);

    for (gint_t i = 0, k = 0, l = 0;i < A.dimension();i++)
    {
        if (k < ndim_AB && idx_A[i] == idx_AB[k])
        {
            len_AB[k] = A.length(i);
            stride_A_AB[k++] = A.stride(i);
        }
        else if (l < ndim_AC && idx_A[i] == idx_AC[l])
        {
            len_AC[l] = A.length(i);
            stride_A_AC[l++] = A.stride(i);
        }
    }

    for (gint_t i = 0, k = 0, l = 0;i < B.dimension();i++)
    {
        if (k < ndim_AB && idx_B[i] == idx_AB[k])
        {
            stride_B_AB[k++] = B.stride(i);
        }
        else if (l < ndim_BC && idx_B[i] == idx_BC[l])
        {
            len_BC[l] = B.length(i);
            stride_B_BC[l++] = B.stride(i);
        }
    }

    for (gint_t i = 0, k = 0, l = 0;i < C.dimension();i++)
    {
        if (k < ndim_AC && idx_C[i] == idx_AC[k])
        {
            stride_C_AC[k++] = C.stride(i);
        }
        else if (l < ndim_BC && idx_C[i] == idx_BC[l])
        {
            stride_C_BC[l++] = C.stride(i);
        }
    }

    Iterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
    Iterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    Iterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);

    const T* restrict A_ = A.data();
    const T* restrict B_ = B.data();
          T* restrict C_ = C.data();

    while (iter_AC.next(A_, C_))
    {
        while (iter_BC.next(B_, C_))
        {
            T temp = T();

            if (alpha != 0.0)
            {
                while (iter_AB.next(A_, B_))
                {
                    assert (A_-A.data() >= 0 && A_-A.data() < A.size());
                    assert (B_-B.data() >= 0 && B_-B.data() < B.size());
                    temp += (*A_)*(*B_);
                }
                temp *= alpha;
            }

            if (beta == 0.0)
            {
                assert (C_-C.data() >= 0 && C_-C.data() < C.size());
                *C_ = temp;
            }
            else if (beta == 1.0)
            {
                assert (C_-C.data() >= 0 && C_-C.data() < C.size());
                *C_ += temp;
            }
            else
            {
                assert (C_-C.data() >= 0 && C_-C.data() < C.size());
                *C_ = temp + beta*(*C_);
            }
        }
    }

    return 0;
}

template
int tensor_contract_reference<   float>(   float alpha, const Tensor<   float>& A, const std::string& idx_A,
                                                        const Tensor<   float>& B, const std::string& idx_B,
                                           float  beta,       Tensor<   float>& C, const std::string& idx_C);

template
int tensor_contract_reference<  double>(  double alpha, const Tensor<  double>& A, const std::string& idx_A,
                                                        const Tensor<  double>& B, const std::string& idx_B,
                                          double  beta,       Tensor<  double>& C, const std::string& idx_C);

template
int tensor_contract_reference<scomplex>(scomplex alpha, const Tensor<scomplex>& A, const std::string& idx_A,
                                                        const Tensor<scomplex>& B, const std::string& idx_B,
                                        scomplex  beta,       Tensor<scomplex>& C, const std::string& idx_C);

template
int tensor_contract_reference<dcomplex>(dcomplex alpha, const Tensor<dcomplex>& A, const std::string& idx_A,
                                                        const Tensor<dcomplex>& B, const std::string& idx_B,
                                        dcomplex  beta,       Tensor<dcomplex>& C, const std::string& idx_C);

}
}
