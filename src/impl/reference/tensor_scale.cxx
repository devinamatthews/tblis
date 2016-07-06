#include "tblis.hpp"

using namespace std;
using namespace MArray;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_scale_reference(T alpha, const tensor_view<T>& A, const std::string& idx_A)
{
    viterator<> iter_A(A.lengths(), A.strides());

    T* restrict A_ = A.data();

    if (alpha == T(0))
    {
        while (iter_A.next(A_))
        {
            *A_ = T();
        }
    }
    else if (alpha == T(0))
    {
        // do nothing
    }
    else
    {
        while (iter_A.next(A_))
        {
            *A_ *= alpha;
        }
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_scale_reference(T alpha, const tensor_view<T>& A, const std::string& idx_A);
#include "tblis_instantiate_for_types.hpp"

}
}
