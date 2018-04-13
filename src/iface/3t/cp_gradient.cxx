#include "cp_gradient.h"

#include "util/macros.h"
#include "util/tensor.hpp"
#include "internal/3t/dense/cp_gradient.hpp"

namespace tblis
{

extern "C"
{

void tblis_tensor_cp_gradient(const tblis_comm* comm, const tblis_config* cfg,
                              tblis_tensor* A, const label_type* idx_A,
                              const tblis_matrix* const * U, const label_type* const * idx_U,
                              tblis_matrix* G, const label_type* idx_G)
{
    unsigned ndim_m = A->ndim;

    for (unsigned i = 0;i < ndim_m;i++)
        TBLIS_ASSERT(A->type == U[i]->type);

    TBLIS_ASSERT(ndim_m >= 2);

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

    unsigned i_r, i_n;
    if (idx_G[0] == idx_r)
    {
        i_r = 0; i_n = 1;
    }
    else if (idx_G[1] == idx_r)
    {
        i_r = 1; i_n = 0;
    }
    else TBLIS_ASSERT(0);

    label_type idx_n = idx_G[i_n];
    len_type len_n = (&G->m)[i_n];
    TBLIS_ASSERT((&G->m)[i_r] == len_r);
    stride_type stride_G_r = (&G->rs)[i_r];
    stride_type stride_G_n = (&G->rs)[i_n];

    for (unsigned i = 0;i < ndim_m;i++)
    {
        if (idx_U_m[i] == idx_n)
        {
            idx_U_m.erase(idx_U_m.begin()+i);
            len_U_m.erase(len_U_m.begin()+i);
            stride_U_m.erase(stride_U_m.begin()+i);
            stride_U_r.erase(stride_U_r.begin()+i);
            break;
        }
    }

    ndim_m = idx_U_m.size();
    TBLIS_ASSERT(ndim_m == A->ndim-1);

    label_vector idx_A_m;
    len_vector len_A_m;
    stride_vector stride_A_m;
    stride_type stride_A_r, stride_A_n;
    for (unsigned i = 0;i < A->ndim;i++)
    {
        if (idx_A[i] == idx_n)
        {
            TBLIS_ASSERT(A->len[i] == len_n);
            stride_A_n = A->stride[i];
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
        ptr_vector<const T> data_U;
        const T* data_A = static_cast<const T*>(A->data);
        T* data_G = static_cast<T*>(G->data);

        for (unsigned i = 0;i < ndim_m+1;i++)
        {
            if (idx_U[i][0] == idx_n || idx_U[i][1] == idx_n) continue;
            data_U.push_back(static_cast<const T*>(U[i]->data));
        }

        data_U = stl_ext::select_from(data_U, idx_U_m, idx_A_m);

        parallelize_if(
        [&](const communicator& comm)
        {
            internal::cp_gradient<T>(comm, get_config(cfg), len_A_m, len_n, len_r,
                                     data_A, stride_A_m, stride_A_n,
                                     data_U, stride_U_m, stride_U_r,
                                     data_G, stride_G_n, stride_G_r);
        }, comm);

        G->alpha<T>() = T(1);
        G->conj = false;
    })
}

}

}
