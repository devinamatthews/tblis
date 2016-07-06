#include "tblis.hpp"

using namespace std;
using namespace MArray;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_dot_reference(const const_tensor_view<T>& A, const std::string& idx_A,
                         const const_tensor_view<T>& B, const std::string& idx_B, T& val)
{
    viterator<2> iter_AB(A.lengths(), A.strides(), B.strides());

    const T* restrict A_ = A.data();
    const T* restrict B_ = B.data();

    val = T(0);

    while (iter_AB.next(A_, B_))
    {
        val += (*A_)*(*B_);
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_dot_reference(const const_tensor_view<T>& A, const std::string& idx_A, \
                         const const_tensor_view<T>& B, const std::string& idx_B, T& val);
#include "tblis_instantiate_for_types.hpp"

}
}
