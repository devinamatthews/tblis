#include "tblis.hpp"

using namespace std;
using namespace stl_ext;
using namespace MArray;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_mult_reference(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                   const const_tensor_view<T>& B, const std::string& idx_B,
                          T  beta, const       tensor_view<T>& C, const std::string& idx_C)
{
    if (alpha == T(0))
    {
        return tensor_scale_reference(beta, C, idx_C);
    }

    string idx_ABC = intersection(idx_A, idx_B, idx_C);
    string idx_AB = exclusion(intersection(idx_A, idx_B), idx_ABC);
    string idx_AC = exclusion(intersection(idx_A, idx_C), idx_ABC);
    string idx_BC = exclusion(intersection(idx_B, idx_C), idx_ABC);
    string idx_A_only = exclusion(idx_A, idx_B, idx_C);
    string idx_B_only = exclusion(idx_B, idx_A, idx_C);
    string idx_C_only = exclusion(idx_C, idx_A, idx_B);

    vector<stride_type> len_A(idx_A_only.size());
    vector<stride_type> len_B(idx_B_only.size());
    vector<stride_type> len_C(idx_C_only.size());
    vector<stride_type> len_AB(idx_AB.size());
    vector<stride_type> len_AC(idx_AC.size());
    vector<stride_type> len_BC(idx_BC.size());
    vector<stride_type> len_ABC(idx_ABC.size());

    vector<stride_type> stride_A_A(idx_A_only.size());
    vector<stride_type> stride_B_B(idx_B_only.size());
    vector<stride_type> stride_C_C(idx_C_only.size());
    vector<stride_type> stride_A_AB(idx_AB.size());
    vector<stride_type> stride_B_AB(idx_AB.size());
    vector<stride_type> stride_A_AC(idx_AC.size());
    vector<stride_type> stride_C_AC(idx_AC.size());
    vector<stride_type> stride_B_BC(idx_BC.size());
    vector<stride_type> stride_C_BC(idx_BC.size());
    vector<stride_type> stride_A_ABC(idx_ABC.size());
    vector<stride_type> stride_B_ABC(idx_ABC.size());
    vector<stride_type> stride_C_ABC(idx_ABC.size());

    for (unsigned i = 0, j = 0, k = 0, l = 0, m = 0;i < A.dimension();i++)
    {
        if (j < idx_A_only.size() && idx_A[i] == idx_A_only[j])
        {
            len_A[j] = A.length(i);
            stride_A_A[j++] = A.stride(i);
        }
        else if (k < idx_AB.size() && idx_A[i] == idx_AB[k])
        {
            len_AB[k] = A.length(i);
            stride_A_AB[k++] = A.stride(i);
        }
        else if (l < idx_AC.size() && idx_A[i] == idx_AC[l])
        {
            len_AC[l] = A.length(i);
            stride_A_AC[l++] = A.stride(i);
        }
        else if (m < idx_ABC.size() && idx_A[i] == idx_ABC[m])
        {
            len_ABC[m] = A.length(i);
            stride_A_ABC[m++] = A.stride(i);
        }
    }

    for (unsigned i = 0, j = 0, k = 0, l = 0, m = 0;i < B.dimension();i++)
    {
        if (j < idx_B_only.size() && idx_B[i] == idx_B_only[j])
        {
            len_B[j] = B.length(i);
            stride_B_B[j++] = B.stride(i);
        }
        else if (k < idx_AB.size() && idx_B[i] == idx_AB[k])
        {
            stride_B_AB[k++] = B.stride(i);
        }
        else if (l < idx_BC.size() && idx_B[i] == idx_BC[l])
        {
            len_BC[l] = B.length(i);
            stride_B_BC[l++] = B.stride(i);
        }
        else if (m < idx_ABC.size() && idx_B[i] == idx_ABC[m])
        {
            stride_B_ABC[m++] = B.stride(i);
        }
    }

    for (unsigned i = 0, j = 0, k = 0, l = 0, m = 0;i < C.dimension();i++)
    {
        if (j < idx_C_only.size() && idx_C[i] == idx_C_only[j])
        {
            len_C[j] = C.length(i);
            stride_C_C[j++] = C.stride(i);
        }
        else if (k < idx_AC.size() && idx_C[i] == idx_AC[k])
        {
            stride_C_AC[k++] = C.stride(i);
        }
        else if (l < idx_BC.size() && idx_C[i] == idx_BC[l])
        {
            stride_C_BC[l++] = C.stride(i);
        }
        else if (m < idx_ABC.size() && idx_C[i] == idx_ABC[m])
        {
            stride_C_ABC[m++] = C.stride(i);
        }
    }

    viterator<1> iter_A(len_A, stride_A_A);
    viterator<1> iter_B(len_B, stride_B_B);
    viterator<1> iter_C(len_C, stride_C_C);
    viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
    viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);
    viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

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

                while (iter_AB.next(A_, B_))
                {
                    T temp_A = T();
                    while (iter_A.next(A_))
                    {
                        temp_A += *A_;
                    }

                    T temp_B = T();
                    while (iter_B.next(B_))
                    {
                        temp_B += *B_;
                    }

                    temp += temp_A*temp_B;
                }

                temp *= alpha;

                if (beta == T(0))
                {
                    while (iter_C.next(C_))
                    {
                        *C_ = temp;
                    }
                }
                else if (beta == T(1))
                {
                    while (iter_C.next(C_))
                    {
                        *C_ += temp;
                    }
                }
                else
                {
                    while (iter_C.next(C_))
                    {
                        *C_ = temp + beta*(*C_);
                    }
                }
            }
        }
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_mult_reference<T>(T alpha, const const_tensor_view<T>& A, const std::string& idx_A, \
                                      const const_tensor_view<T>& B, const std::string& idx_B, \
                             T  beta, const       tensor_view<T>& C, const std::string& idx_C);
#include "tblis_instantiate_for_types.hpp"

}
}
