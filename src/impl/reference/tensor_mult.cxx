#include "tblis.hpp"
#include "impl/tensor_impl.hpp"

using namespace std;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_mult_reference(T alpha, const Tensor<T>& A, const std::string& idx_A,
                                   const Tensor<T>& B, const std::string& idx_B,
                          T  beta,       Tensor<T>& C, const std::string& idx_C)
{
    string idx_A_only, idx_B_only, idx_C_only;
    string idx_AB, idx_AC, idx_BC;
    string idx_ABC;

    util::set_intersection(idx_A, idx_B, idx_AB);
    util::set_intersection(idx_A, idx_C, idx_AC);
    util::set_intersection(idx_B, idx_C, idx_BC);

    gint_t ndim_ABC = util::set_intersection(idx_AB, idx_BC, idx_ABC).size();

    gint_t ndim_A = util::set_difference(
                    util::set_difference(idx_A, idx_AB, idx_A_only), idx_AC).size();
    gint_t ndim_B = util::set_difference(
                    util::set_difference(idx_B, idx_AB, idx_B_only), idx_BC).size();
    gint_t ndim_C = util::set_difference(
                    util::set_difference(idx_C, idx_AC, idx_C_only), idx_BC).size();

    gint_t ndim_AB = util::set_difference(idx_AB, idx_ABC).size();
    gint_t ndim_AC = util::set_difference(idx_AC, idx_ABC).size();
    gint_t ndim_BC = util::set_difference(idx_BC, idx_ABC).size();

    vector<inc_t> len_A(ndim_A);
    vector<inc_t> len_B(ndim_B);
    vector<inc_t> len_C(ndim_C);
    vector<inc_t> len_AB(ndim_AB);
    vector<inc_t> len_AC(ndim_AC);
    vector<inc_t> len_BC(ndim_BC);
    vector<inc_t> len_ABC(ndim_ABC);

    vector<inc_t> stride_A_A(ndim_A);
    vector<inc_t> stride_B_B(ndim_B);
    vector<inc_t> stride_C_C(ndim_C);
    vector<inc_t> stride_A_AB(ndim_AB);
    vector<inc_t> stride_B_AB(ndim_AB);
    vector<inc_t> stride_A_AC(ndim_AC);
    vector<inc_t> stride_C_AC(ndim_AC);
    vector<inc_t> stride_B_BC(ndim_BC);
    vector<inc_t> stride_C_BC(ndim_BC);
    vector<inc_t> stride_A_ABC(ndim_ABC);
    vector<inc_t> stride_B_ABC(ndim_ABC);
    vector<inc_t> stride_C_ABC(ndim_ABC);

    for (gint_t i = 0, j = 0, k = 0, l = 0, m = 0;i < A.dimension();i++)
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
        else if (l < ndim_AC && idx_A[i] == idx_AC[l])
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

    for (gint_t i = 0, j = 0, k = 0, l = 0, m = 0;i < B.dimension();i++)
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
        else if (l < ndim_BC && idx_B[i] == idx_BC[l])
        {
            len_BC[l] = B.length(i);
            stride_B_BC[l++] = B.stride(i);
        }
        else if (m < ndim_ABC && idx_B[i] == idx_ABC[m])
        {
            stride_B_ABC[m++] = B.stride(i);
        }
    }

    for (gint_t i = 0, j = 0, k = 0, l = 0, m = 0;i < C.dimension();i++)
    {
        if (j < ndim_C && idx_C[i] == idx_C_only[j])
        {
            len_C[j] = C.length(i);
            stride_C_C[j++] = C.stride(i);
        }
        else if (k < ndim_AC && idx_C[i] == idx_AC[k])
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

    Iterator<1> iter_A(len_A, stride_A_A);
    Iterator<1> iter_B(len_B, stride_B_B);
    Iterator<1> iter_C(len_C, stride_C_C);
    Iterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
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
            while (iter_BC.next(B_, C_))
            {
                T temp = T();

                if (alpha != 0.0)
                {
                    while (iter_AB.next(A_, B_))
                    {
                        T temp_A = T();
                        while (iter_A.next(A_))
                        {
                            assert (A_-A.data() >= 0 && A_-A.data() < A.size());
                            temp_A += *A_;
                        }

                        T temp_B = T();
                        while (iter_B.next(B_))
                        {
                            assert (B_-B.data() >= 0 && B_-B.data() < B.size());
                            temp_B += *B_;
                        }

                        temp += temp_A*temp_B;
                    }

                    temp *= alpha;
                }

                if (beta == 0.0)
                {
                    while (iter_C.next(C_))
                    {
                        assert (C_-C.data() >= 0 && C_-C.data() < C.size());
                        *C_ = temp;
                    }
                }
                else if (beta == 1.0)
                {
                    while (iter_C.next(C_))
                    {
                        assert (C_-C.data() >= 0 && C_-C.data() < C.size());
                        *C_ += temp;
                    }
                }
                else
                {
                    while (iter_C.next(C_))
                    {
                        assert (C_-C.data() >= 0 && C_-C.data() < C.size());
                        *C_ = temp + beta*(*C_);
                    }
                }
            }
        }
    }

    return 0;
}

template
int tensor_mult_reference<   float>(   float alpha, const Tensor<   float>& A, const std::string& idx_A,
                                                    const Tensor<   float>& B, const std::string& idx_B,
                                       float  beta,       Tensor<   float>& C, const std::string& idx_C);

template
int tensor_mult_reference<  double>(  double alpha, const Tensor<  double>& A, const std::string& idx_A,
                                                    const Tensor<  double>& B, const std::string& idx_B,
                                      double  beta,       Tensor<  double>& C, const std::string& idx_C);

template
int tensor_mult_reference<sComplex>(sComplex alpha, const Tensor<sComplex>& A, const std::string& idx_A,
                                                    const Tensor<sComplex>& B, const std::string& idx_B,
                                    sComplex  beta,       Tensor<sComplex>& C, const std::string& idx_C);

template
int tensor_mult_reference<dComplex>(dComplex alpha, const Tensor<dComplex>& A, const std::string& idx_A,
                                                    const Tensor<dComplex>& B, const std::string& idx_B,
                                    dComplex  beta,       Tensor<dComplex>& C, const std::string& idx_C);

}
}
