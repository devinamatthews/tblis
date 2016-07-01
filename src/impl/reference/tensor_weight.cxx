#include "tblis.hpp"
#include "impl/tensor_impl.hpp"

using namespace std;
using namespace stl_ext;
using namespace MArray;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_weight_reference(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                     const const_tensor_view<T>& B, const std::string& idx_B,
                            T  beta,             tensor_view<T>& C, const std::string& idx_C)
{
    string idx_ABC = intersection(idx_A, idx_B, idx_C);
    string idx_AC = exclusion(intersection(idx_A, idx_C), idx_ABC);
    string idx_BC = exclusion(intersection(idx_B, idx_C), idx_ABC);

    vector<stride_type> len_AC(idx_AC.size());
    vector<stride_type> len_BC(idx_BC.size());
    vector<stride_type> len_ABC(idx_ABC.size());

    vector<stride_type> stride_A_A(idx_A.size());
    vector<stride_type> stride_B_B(idx_B.size());
    vector<stride_type> stride_C_C(idx_C.size());
    vector<stride_type> stride_A_AC(idx_AC.size());
    vector<stride_type> stride_C_AC(idx_AC.size());
    vector<stride_type> stride_B_BC(idx_BC.size());
    vector<stride_type> stride_C_BC(idx_BC.size());
    vector<stride_type> stride_A_ABC(idx_ABC.size());
    vector<stride_type> stride_B_ABC(idx_ABC.size());
    vector<stride_type> stride_C_ABC(idx_ABC.size());

    for (unsigned i = 0, l = 0, m = 0;i < A.dimension();i++)
    {
        if (l < idx_AC.size() && idx_A[i] == idx_AC[l])
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

    for (unsigned i = 0, l = 0, m = 0;i < B.dimension();i++)
    {
        if (l < idx_BC.size() && idx_B[i] == idx_BC[l])
        {
            len_BC[l] = B.length(i);
            stride_B_BC[l++] = B.stride(i);
        }
        else if (m < idx_ABC.size() && idx_B[i] == idx_ABC[m])
        {
            stride_B_ABC[m++] = B.stride(i);
        }
    }

    for (unsigned i = 0, k = 0, l = 0, m = 0;i < C.dimension();i++)
    {
        if (k < idx_AC.size() && idx_C[i] == idx_AC[k])
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

    const T* restrict A_ = A.data();
    const T* restrict B_ = B.data();
          T* restrict C_ = C.data();

    if (alpha == 0.0)
    {
        viterator<1> iter_C(C.lengths(), C.strides());

        if (beta == 0.0)
        {
            while (iter_C.next(C_))
            {
                *C_ = T();
            }
        }
        else
        {
            while (iter_C.next(C_))
            {
                *C_ *= beta;
            }
        }
    }
    else
    {
        viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
        viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);
        viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

        while (iter_ABC.next(A_, B_, C_))
        {
            while (iter_AC.next(A_, C_))
            {
                if (beta == 0.0)
                {
                    while (iter_BC.next(B_, C_))
                    {
                        *C_ = alpha*(*A_)*(*B_);
                    }
                }
                else
                {
                    while (iter_BC.next(B_, C_))
                    {
                        *C_ = alpha*(*A_)*(*B_) + beta*(*C_);
                    }
                }
            }
        }
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_weight_reference<T>(T alpha, const const_tensor_view<T>& A, const std::string& idx_A, \
                                        const const_tensor_view<T>& B, const std::string& idx_B, \
                               T  beta,             tensor_view<T>& C, const std::string& idx_C);
#include "tblis_instantiate_for_types.hpp"

}
}
