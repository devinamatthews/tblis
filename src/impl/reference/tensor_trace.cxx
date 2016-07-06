#include "tblis.hpp"

using namespace std;
using namespace stl_ext;
using namespace MArray;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_trace_reference(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                           T  beta, const       tensor_view<T>& B, const std::string& idx_B)
{
    string idx_AB = intersection(idx_A, idx_B);
    string idx_A_only = exclusion(idx_A, idx_AB);

    vector<idx_type> len_A(idx_A_only.size());
    vector<idx_type> len_AB(idx_AB.size());

    vector<stride_type> stride_A_A(idx_A_only.size());
    vector<stride_type> stride_A_AB(idx_AB.size());
    vector<stride_type> stride_B_AB(idx_AB.size());

    for (unsigned i = 0;i < idx_A_only.size();i++)
        for (unsigned j = 0;j < A.dimension();j++)
            if (idx_A_only[i] == idx_A[j])
            {
                len_A[i] = A.length(j);
                stride_A_A[i] = A.stride(j);
            }

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

    viterator<1> iter_A(len_A, stride_A_A);
    viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);

    const T* restrict A_ = A.data();
          T* restrict B_ = B.data();

    while (iter_AB.next(A_, B_))
    {
        T temp = T();

        if (alpha != T(0))
        {
            while (iter_A.next(A_))
            {
                temp += *A_;
            }
            temp *= alpha;
        }

        if (beta == T(0))
        {
            *B_ = temp;
        }
        else if (beta == T(1))
        {
            *B_ += temp;
        }
        else
        {
            *B_ = temp + beta*(*B_);
        }
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_trace_reference<T>(T alpha, const const_tensor_view<T>& A, const std::string& idx_A, \
                              T  beta, const       tensor_view<T>& B, const std::string& idx_B);
#include "tblis_instantiate_for_types.hpp"

}
}
