#include "tblis_reduce.hpp"

#include <limits>

#include "tblis_config.hpp"

namespace tblis
{

template <typename T>
int tensor_reduce_ref(reduce_t op, const std::vector<len_type>& len_A,
                      const T* TBLIS_RESTRICT A, const std::vector<stride_type>& stride_A,
                      T& val, stride_type& idx)
{
    MArray::viterator<> iter_A(len_A, stride_A);

    const T* const A0 = A;

    switch (op)
    {
        case REDUCE_SUM:
        case REDUCE_SUM_ABS:
        case REDUCE_NORM_2:
            val = T();
            break;
        case REDUCE_MAX:
        case REDUCE_MAX_ABS:
            val = std::numeric_limits<real_type_t<T>>::lowest();
            break;
        case REDUCE_MIN:
        case REDUCE_MIN_ABS:
            val = std::numeric_limits<real_type_t<T>>::max();
            break;
    }

    idx = 0;

    switch (op)
    {
        case REDUCE_SUM:
            while (iter_A.next(A))
            {
                val += *A;
            }
            break;
        case REDUCE_SUM_ABS:
            while (iter_A.next(A))
            {
                val += std::abs(*A);
            }
            break;
        case REDUCE_MAX:
            while (iter_A.next(A))
            {
                if (*A > val)
                {
                    val = *A;
                    idx = A-A0;
                }
            }
            break;
        case REDUCE_MAX_ABS:
            while (iter_A.next(A))
            {
                if (std::abs(*A) > val)
                {
                    val = std::abs(*A);
                    idx = A-A0;
                }
            }
            break;
        case REDUCE_MIN:
            while (iter_A.next(A))
            {
                if (*A < val)
                {
                    val = *A;
                    idx = A-A0;
                }
            }
            break;
        case REDUCE_MIN_ABS:
            while (iter_A.next(A))
            {
                if (std::abs(*A) < val)
                {
                    val = std::abs(*A);
                    idx = A-A0;
                }
            }
            break;
        case REDUCE_NORM_2:
            while (iter_A.next(A))
            {
                val += norm2(*A);
            }
            val = sqrt(real(val));
            break;
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_reduce_ref(reduce_t op, const std::vector<idx_type>& len_A, \
                      const T* A, const std::vector<stride_type>& stride_A, \
                      T& val, stride_type& idx);
#include "tblis_instantiate_for_types.hpp"

}
