#include "hosvd.h"

#include "util/macros.h"
#include "util/tensor.hpp"
#include "internal/3t/dense/hosvd.hpp"

namespace tblis
{

extern "C"
{

void tblis_tensor_hosvd(tblis_tensor* A, tblis_matrix* const * U,
                        double tol, bool iterate)
{
    unsigned ndim = A->ndim;

    len_vector len_A(A->len, A->len+ndim);
    len_vector len_B(ndim);
    stride_vector stride_A(A->stride, A->stride+ndim);
    stride_vector ld_U(ndim);

    for (unsigned i = 0;i < ndim;i++)
    {
        TBLIS_ASSERT(U[i]->m == len_A[i]);
        TBLIS_ASSERT(U[i]->n == len_A[i]);
        TBLIS_ASSERT(U[i]->rs == 1);
        ld_U[i] = U[i]->cs;
    }

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        T* data_A = static_cast<const T*>(A->data);

        ptr_vector<T> data_U;
        for (unsigned i = 0;i < ndim;i++)
            data_U.push_back(static_cast<T*>(U[i]->data));

        internal::hosvd<T>(len_A, data_A, stride_A,
                           data_U, ld_U, tol, iterate);

        A->alpha<T>() = T(1);
        A->conj = false;

        for (unsigned i = 0;i < ndim;i++)
        {
            U[i]->alpha<T>() = T(1);
            U[i]->conj = false;
            U[i]->n = len_B[i];
        }
    })
}

}

}
