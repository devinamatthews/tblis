#include "tblis.hpp"
#include "impl/tensor_impl.hpp"

using namespace std;
using namespace MArray;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_reduce_reference(reduce_t op, const const_tensor_view<T>& A, const std::string& idx_A, T& val, stride_type& idx)
{
    viterator<> iter_A(A.lengths(), A.strides());

    const T* restrict A_  = A.data();
    const T* const    A0_ = A_;

    switch (op)
    {
        case REDUCE_SUM:
        case REDUCE_SUM_ABS:
        case REDUCE_NORM_2:
            val = 0.0;
            break;
        case REDUCE_MAX:
        case REDUCE_MAX_ABS:
            val = numeric_limits<typename real_type<T>::type>::lowest();
            break;
        case REDUCE_MIN:
        case REDUCE_MIN_ABS:
            val = numeric_limits<typename real_type<T>::type>::max();
            break;
    }

    idx = 0;

    switch (op)
    {
        case REDUCE_SUM:
            for (;iter_A.next(A_);)
            {
                assert (A_-A.data() >= 0 && A_-A.data() < A.size());
                val += *A_;
            }
            break;
        case REDUCE_SUM_ABS:
            for (;iter_A.next(A_);)
            {
                assert (A_-A.data() >= 0 && A_-A.data() < A.size());
                val += std::abs(*A_);
            }
            break;
        case REDUCE_MAX:
            for (;iter_A.next(A_);)
            {
                assert (A_-A.data() >= 0 && A_-A.data() < A.size());
                if (*A_ > val)
                {
                    val = *A_;
                    idx = A_-A0_;
                }
            }
            break;
        case REDUCE_MAX_ABS:
            for (;iter_A.next(A_);)
            {
                assert (A_-A.data() >= 0 && A_-A.data() < A.size());
                if (std::abs(*A_) > val)
                {
                    val = std::abs(*A_);
                    idx = A_-A0_;
                }
            }
            break;
        case REDUCE_MIN:
            for (;iter_A.next(A_);)
            {
                assert (A_-A.data() >= 0 && A_-A.data() < A.size());
                if (*A_ < val)
                {
                    val = *A_;
                    idx = A_-A0_;
                }
            }
            break;
        case REDUCE_MIN_ABS:
            for (;iter_A.next(A_);)
            {
                assert (A_-A.data() >= 0 && A_-A.data() < A.size());
                if (std::abs(*A_) < val)
                {
                    val = std::abs(*A_);
                    idx = A_-A0_;
                }
            }
            break;
        case REDUCE_NORM_2:
            for (;iter_A.next(A_);)
            {
                assert (A_-A.data() >= 0 && A_-A.data() < A.size());
                val += norm2(*A_);
            }
            val = sqrt(real(val));
            break;
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_reduce_reference<T>(reduce_t op, const const_tensor_view<T>& A, const std::string& idx_A, T& val, stride_type& idx);
#include "tblis_instantiate_for_types.hpp"

}
}
