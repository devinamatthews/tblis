#include "tblis.hpp"
#include "impl/tensor_impl.hpp"

using namespace std;
using namespace MArray;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_scale_reference(T alpha, tensor_view<T>& A, const std::string& idx_A)
{
    viterator<> iter_A(A.lengths(), A.strides());

    T* restrict A_ = A.data();

    if (alpha == 0.0)
    {
        while (iter_A.next(A_))
        {
            assert (A_-A.data() >= 0 && A_-A.data() < A.size());
            *A_ = 0.0;
        }
    }
    else if (alpha == 1.0)
    {
        // do nothing
    }
    else
    {
        while (iter_A.next(A_))
        {
            assert (A_-A.data() >= 0 && A_-A.data() < A.size());
            *A_ *= alpha;
        }
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_scale_reference(T alpha, tensor_view<T>& A, const std::string& idx_A);
#include "tblis_instantiate_for_types.hpp"

}
}
