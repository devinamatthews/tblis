#include "tblis.hpp"

using namespace std;
using namespace stl_ext;
using namespace MArray;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_outer_prod_reference(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                         const const_tensor_view<T>& B, const std::string& idx_B,
                                T  beta, const       tensor_view<T>& C, const std::string& idx_C)
{
    if (alpha == T(0))
    {
        return tensor_scale_reference(beta, C, idx_C);
    }

    string idx_AC = intersection(idx_A, idx_C);
    string idx_BC = intersection(idx_B, idx_C);

    vector<stride_type> len_AC(idx_AC.size());
    vector<stride_type> len_BC(idx_BC.size());

    vector<stride_type> stride_A_AC(idx_AC.size());
    vector<stride_type> stride_C_AC(idx_AC.size());
    vector<stride_type> stride_B_BC(idx_BC.size());
    vector<stride_type> stride_C_BC(idx_BC.size());

    for (unsigned i = 0;i < idx_AC.size();i++)
        for (unsigned j = 0;j < A.dimension();j++)
            if (idx_AC[i] == idx_A[j])
            {
                len_AC[i] = A.length(j);
                stride_A_AC[i] = A.stride(j);
            }

    for (unsigned i = 0;i < idx_AC.size();i++)
        for (unsigned j = 0;j < C.dimension();j++)
            if (idx_AC[i] == idx_C[j]) stride_C_AC[i] = C.stride(j);

    for (unsigned i = 0;i < idx_BC.size();i++)
        for (unsigned j = 0;j < B.dimension();j++)
            if (idx_BC[i] == idx_B[j])
            {
                len_BC[i] = B.length(j);
                stride_B_BC[i] = B.stride(j);
            }

    for (unsigned i = 0;i < idx_BC.size();i++)
        for (unsigned j = 0;j < C.dimension();j++)
            if (idx_BC[i] == idx_C[j]) stride_C_BC[i] = C.stride(j);

    const T* restrict A_ = A.data();
    const T* restrict B_ = B.data();
          T* restrict C_ = C.data();

    viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);

    while (iter_AC.next(A_, C_))
    {
        if (beta == T(0))
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

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_outer_prod_reference(T alpha, const const_tensor_view<T>& A, const std::string& idx_A, \
                                         const const_tensor_view<T>& B, const std::string& idx_B, \
                                T  beta, const       tensor_view<T>& C, const std::string& idx_C);
#include "tblis_instantiate_for_types.hpp"

}
}
