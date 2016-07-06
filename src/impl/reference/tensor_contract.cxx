#include "tblis.hpp"

using namespace std;
using namespace stl_ext;
using namespace MArray;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_contract_reference(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                       const const_tensor_view<T>& B, const std::string& idx_B,
                              T  beta, const       tensor_view<T>& C, const std::string& idx_C)
{
    if (alpha == T(0))
    {
        return tensor_scale_reference(beta, C, idx_C);
    }

    string idx_AB = intersection(idx_A, idx_B);
    string idx_AC = intersection(idx_A, idx_C);
    string idx_BC = intersection(idx_B, idx_C);

    vector<stride_type> len_AB(idx_AB.size());
    vector<stride_type> len_AC(idx_AC.size());
    vector<stride_type> len_BC(idx_BC.size());

    vector<stride_type> stride_A_AB(idx_AB.size());
    vector<stride_type> stride_B_AB(idx_AB.size());
    vector<stride_type> stride_A_AC(idx_AC.size());
    vector<stride_type> stride_C_AC(idx_AC.size());
    vector<stride_type> stride_B_BC(idx_BC.size());
    vector<stride_type> stride_C_BC(idx_BC.size());

    for (unsigned i = 0;i < idx_AB.size();i++)
        for (unsigned j = 0;j < A.dimension();j++)
            if (idx_AB[i] == idx_A[j])
            {
                len_AB[i] = A.length(j);
                stride_A_AB[i] = A.stride(j);
            }

    for (unsigned i = 0;i < idx_AB.size();i++)
        for (unsigned j = 0;j < B.dimension();j++)
            if (idx_AB[i] == idx_B[j]) stride_B_AB[i] = B.stride(j);

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

    viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
    viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);

    const T* restrict A_ = A.data();
    const T* restrict B_ = B.data();
          T* restrict C_ = C.data();

    while (iter_AC.next(A_, C_))
    {
        while (iter_BC.next(B_, C_))
        {
            T temp = T();

            while (iter_AB.next(A_, B_))
            {
                temp += (*A_)*(*B_);
            }
            temp *= alpha;

            if (beta == T(0))
            {
                *C_ = temp;
            }
            else if (beta == T(1))
            {
                *C_ += temp;
            }
            else
            {
                *C_ = temp + beta*(*C_);
            }
        }
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_contract_reference(T alpha, const const_tensor_view<T>& A, const std::string& idx_A, \
                                       const const_tensor_view<T>& B, const std::string& idx_B, \
                              T  beta, const       tensor_view<T>& C, const std::string& idx_C);
#include "tblis_instantiate_for_types.hpp"

}
}
