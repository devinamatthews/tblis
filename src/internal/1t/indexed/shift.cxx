#include "shift.hpp"
#include "internal/1t/dense/scale.hpp"
#include "internal/1t/dense/set.hpp"
#include "internal/1t/dense/shift.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void shift(const communicator& comm, const config& cfg,
           T alpha, T beta, bool conj_A, const indexed_varray_view<T>& A,
           const dim_vector& idx_A_A)
{
    for (len_type i = 0;i < A.num_indices();i++)
    {
        if (A.factor(i) == T(0))
        {
            if (beta == T(0))
            {
                set(comm, cfg, A.dense_lengths(),
                    T(0), A.data(i), A.dense_strides());
            }
            else if (beta != T(1) || (is_complex<T>::value && conj_A))
            {
                scale(comm, cfg, A.dense_lengths(),
                      beta, conj_A, A.data(i), A.dense_strides());
            }
        }
        else
        {
            shift(comm, cfg, A.dense_lengths(),
                  A.factor(i)*alpha, beta, conj_A, A.data(i), A.dense_strides());
        }
    }
}

#define FOREACH_TYPE(T) \
template void shift(const communicator& comm, const config& cfg, \
                    T alpha, T beta, bool conj_A, const indexed_varray_view<T>& A, \
                    const dim_vector&);
#include "configs/foreach_type.h"

}
}
