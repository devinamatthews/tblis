#include "scale.h"

#include "util/macros.h"
#include "util/tensor.hpp"
#include "internal/1t/scale.hpp"
#include "internal/1t/set.hpp"

namespace tblis
{

extern "C"
{

void tblis_tensor_scale(const tblis_comm* comm, const tblis_config* cfg,
                        tblis_tensor* A, const label_type* idx_A_)
{
    unsigned ndim_A = A->ndim;
    std::vector<len_type> len_A;
    std::vector<stride_type> stride_A;
    std::vector<label_type> idx_A;
    diagonal(ndim_A, A->len, A->stride, idx_A_, len_A, stride_A, idx_A);

    fold(len_A, idx_A, stride_A);

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        if (A->alpha<T>() == T(0))
        {
            parallelize_if(internal::set<T>, comm, get_config(cfg), len_A,
                           T(0), static_cast<T*>(A->data), stride_A);
        }
        else if (A->alpha<T>() != T(1))
        {
            parallelize_if(internal::scale<T>, comm, get_config(cfg), len_A,
                           A->alpha<T>(), A->conj, static_cast<T*>(A->data), stride_A);
        }

        A->alpha<T>() = T(1);
        A->conj = false;
    })
}

}

}
