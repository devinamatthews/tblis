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
int tensor_replicate_reference(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                               T  beta,             tensor_view<T>& B, const std::string& idx_B)
{
    string idx_AB = intersection(idx_A, idx_B);
    string idx_B_only = exclusion(idx_B, idx_AB);

    vector<idx_type> len_B(idx_B_only.size());
    vector<idx_type> len_AB(idx_AB.size());

    vector<stride_type> stride_B_B(idx_B_only.size());
    vector<stride_type> stride_A_AB(idx_AB.size());
    vector<stride_type> stride_B_AB(idx_AB.size());

    for (unsigned i = 0;i < idx_B_only.size();i++)
        for (unsigned j = 0;j < B.dimension();j++)
            if (idx_B_only[i] == idx_B[j])
            {
                len_B[i] = B.length(j);
                stride_B_B[i] = B.stride(j);
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

    viterator<1> iter_B(len_B, stride_B_B);
    viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);

    const T* restrict A_ = A.data();
          T* restrict B_ = B.data();

    while (iter_AB.next(A_, B_))
    {
        T temp = (alpha == 0.0 ? *A_ : alpha*(*A_));

        if (beta == 0.0)
        {
            while (iter_B.next(B_))
            {
                assert (B_-B.data() >= 0 && B_-B.data() < B.size());
                *B_ = temp;
            }
        }
        else if (beta == 1.0)
        {
            while (iter_B.next(B_))
            {
                assert (B_-B.data() >= 0 && B_-B.data() < B.size());
                *B_ += temp;
            }
        }
        else
        {
            while (iter_B.next(B_))
            {
                assert (B_-B.data() >= 0 && B_-B.data() < B.size());
                *B_ = temp + beta*(*B_);
            }
        }
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_replicate_reference<T>(T alpha, const const_tensor_view<T>& A, const std::string& idx_A, \
                                  T  beta,             tensor_view<T>& B, const std::string& idx_B);
#include "tblis_instantiate_for_types.hpp"

}
}
