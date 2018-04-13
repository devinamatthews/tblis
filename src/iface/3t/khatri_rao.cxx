#include "khatri_rao.h"
#include "mult.h"
#include "../1t/add.h"

#include "util/macros.h"
#include "util/tensor.hpp"
#include "internal/1t/dense/add.hpp"
#include "internal/1t/dense/scale.hpp"
#include "internal/1t/dense/set.hpp"
#include "internal/3t/dense/khatri_rao.hpp"

namespace tblis
{

extern "C"
{

void tblis_tensor_khatri_rao(const tblis_comm* comm, const tblis_config* cfg,
                             const tblis_matrix* const * U, const label_type* const * idx_U,
                             tblis_tensor* A, const label_type* idx_A)
{
    unsigned ndim_m = A->ndim-1;

    for (unsigned i = 0;i < ndim_m;i++)
        TBLIS_ASSERT(A->type == U[i]->type);

    if (ndim_m == 1)
    {
        tblis_tensor Um;
        Um.type = U[0]->type;
        Um.conj = U[0]->conj;
        Um.scalar = U[0]->scalar;
        Um.data = const_cast<void*>(U[0]->data);
        Um.ndim = 2;
        Um.len = const_cast<len_type*>(&U[0]->m);
        Um.stride = const_cast<stride_type*>(&U[0]->rs);
        tblis_tensor_add(comm, cfg, &Um, idx_U[0], A, idx_A);
        return;
    }
    else if (ndim_m == 2)
    {
        tblis_tensor Um0, Um1;
        Um0.type = U[0]->type;
        Um1.type = U[1]->type;
        Um0.conj = U[0]->conj;
        Um1.conj = U[1]->conj;
        Um0.scalar = U[0]->scalar;
        Um1.scalar = U[1]->scalar;
        Um0.data = const_cast<void*>(U[0]->data);
        Um1.data = const_cast<void*>(U[1]->data);
        Um0.ndim = 2;
        Um1.ndim = 2;
        Um0.len = const_cast<len_type*>(&U[0]->m);
        Um1.len = const_cast<len_type*>(&U[1]->m);
        Um0.stride = const_cast<stride_type*>(&U[0]->rs);
        Um1.stride = const_cast<stride_type*>(&U[1]->rs);
        tblis_tensor_mult(comm, cfg, &Um0, idx_U[0], &Um1, idx_U[1], A, idx_A);
        return;
    }

    unsigned i_r_0, i_r_1, i_m_0, i_m_1;
    if (idx_U[0][0] == idx_U[1][0])
    {
        i_r_0 = 0; i_m_0 = 1;
        i_r_1 = 0; i_m_1 = 1;
    }
    else if (idx_U[0][0] == idx_U[1][1])
    {
        i_r_0 = 0; i_m_0 = 1;
        i_r_1 = 1; i_m_1 = 0;
    }
    else if (idx_U[0][1] == idx_U[1][0])
    {
        i_r_0 = 1; i_m_0 = 0;
        i_r_1 = 0; i_m_1 = 1;
    }
    else if (idx_U[0][1] == idx_U[1][1])
    {
        i_r_0 = 1; i_m_0 = 0;
        i_r_1 = 1; i_m_1 = 0;
    }
    else TBLIS_ASSERT(0);

    label_type idx_r = idx_U[0][i_r_0];
    len_type len_r = (&U[0]->m)[i_r_0];
    TBLIS_ASSERT((&U[1]->m)[i_r_1] == len_r);

    label_vector idx_U_m(ndim_m, 0);
    len_vector len_U_m(ndim_m);
    stride_vector stride_U_m(ndim_m);
    stride_vector stride_U_r(ndim_m);

    idx_U_m[0] = idx_U[0][i_m_0];
    idx_U_m[1] = idx_U[1][i_m_1];
    len_U_m[0] = (&U[0]->m)[i_m_0];
    len_U_m[1] = (&U[1]->m)[i_m_1];
    stride_U_m[0] = (&U[0]->rs)[i_m_0];
    stride_U_m[1] = (&U[1]->rs)[i_m_1];
    stride_U_r[0] = (&U[0]->rs)[i_r_0];
    stride_U_r[1] = (&U[1]->rs)[i_r_1];

    TBLIS_ASSERT(idx_U_m[0] != idx_U_m[1]);

    for (unsigned i = 2;i < ndim_m;i++)
    {
        unsigned i_r, i_m;
        if (idx_U[i][0] == idx_r)
        {
            i_r = 0; i_m = 1;
        }
        else if (idx_U[i][1] == idx_r)
        {
            i_r = 1; i_m = 0;
        }
        else TBLIS_ASSERT(0);

        TBLIS_ASSERT((&U[i]->m)[i_r] == len_r);

        idx_U_m[i] = idx_U[i][i_m];
        len_U_m[i] = (&U[i]->m)[i_m];
        stride_U_m[i] = (&U[i]->rs)[i_m];
        stride_U_r[i] = (&U[i]->rs)[i_r];

        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_U_m[i] != idx_U_m[j]);
    }

    label_vector idx_A_m;
    len_vector len_A_m;
    stride_vector stride_A_m;
    stride_type stride_A_r;
    for (unsigned i = 0;i < A->ndim;i++)
    {
        if (idx_A[i] == idx_r)
        {
            TBLIS_ASSERT(A->len[i] == len_r);
            stride_A_r = A->stride[i];
        }
        else
        {
            idx_A_m.push_back(idx_A[i]);
            len_A_m.push_back(A->len[i]);
            stride_A_m.push_back(A->stride[i]);
        }
    }

    TBLIS_ASSERT(idx_A_m.size() == ndim_m);
    TBLIS_ASSERT(stl_ext::sorted(idx_U_m) == stl_ext::sorted(idx_A_m));
    TBLIS_ASSERT(stl_ext::select_from(len_U_m, idx_U_m, idx_A_m) == len_A_m);

    stride_U_m = stl_ext::select_from(stride_U_m, idx_U_m, idx_A_m);
    stride_U_r = stl_ext::select_from(stride_U_r, idx_U_m, idx_A_m);

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        T alpha = T(1);
        T beta = A->alpha<T>();

        ptr_vector<const T> data_U(ndim_m);
        T* data_A = static_cast<T*>(A->data);

        for (unsigned i = 0;i < ndim_m;i++)
        {
            alpha *= U[i]->alpha<T>();
            data_U[i] = static_cast<const T*>(U[i]->data);
        }

        data_U = stl_ext::select_from(data_U, idx_U_m, idx_A_m);

        parallelize_if(
        [&](const communicator& comm)
        {
            if (alpha == T(0))
            {
                len_A_m.push_back(len_r);
                stride_A_m.push_back(stride_A_r);

                if (beta == T(0))
                {
                    internal::set<T>(comm, get_config(cfg), len_A_m,
                                     T(0), data_A, stride_A_m);
                }
                else if (beta != T(1) || (is_complex<T>::value && A->conj))
                {
                    internal::scale<T>(comm, get_config(cfg), len_A_m,
                                       beta, A->conj, data_A, stride_A_m);
                }
            }
            else
            {
                internal::khatri_rao<T>(comm, get_config(cfg), len_A_m, len_r,
                                        alpha, data_U, stride_U_m, stride_U_r,
                                         beta, data_A, stride_A_m, stride_A_r);
            }
        }, comm);

        A->alpha<T>() = T(1);
        A->conj = false;
    })
}

}

}
