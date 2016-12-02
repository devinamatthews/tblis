#include "mult.h"

#include "util/macros.h"
#include "util/tensor.hpp"
#include "internal/1t/scale.hpp"
#include "internal/1t/set.hpp"
#include "internal/3t/mult.hpp"

namespace tblis
{

extern "C"
{

void tblis_tensor_mult(const tblis_comm* comm, const tblis_config* cfg,
                       const tblis_tensor* A, const label_type* idx_A_,
                       const tblis_tensor* B, const label_type* idx_B_,
                             tblis_tensor* C, const label_type* idx_C_)
{
    TBLIS_ASSERT(A->type == B->type);
    TBLIS_ASSERT(A->type == C->type);

    unsigned ndim_A = A->ndim;
    std::vector<len_type> len_A;
    std::vector<stride_type> stride_A;
    std::vector<label_type> idx_A;
    diagonal(ndim_A, A->len, A->stride, idx_A_, len_A, stride_A, idx_A);

    unsigned ndim_B = B->ndim;
    std::vector<len_type> len_B;
    std::vector<stride_type> stride_B;
    std::vector<label_type> idx_B;
    diagonal(ndim_B, B->len, B->stride, idx_B_, len_B, stride_B, idx_B);

    unsigned ndim_C = C->ndim;
    std::vector<len_type> len_C;
    std::vector<stride_type> stride_C;
    std::vector<label_type> idx_C;
    diagonal(ndim_C, C->len, C->stride, idx_C_, len_C, stride_C, idx_C);

    auto idx_ABC = stl_ext::intersection(idx_A, idx_B, idx_C);
    auto len_ABC = stl_ext::select_from(len_A, idx_A, idx_ABC);
    TBLIS_ASSERT(len_ABC == stl_ext::select_from(len_B, idx_B, idx_ABC));
    TBLIS_ASSERT(len_ABC == stl_ext::select_from(len_C, idx_C, idx_ABC));
    auto stride_A_ABC = stl_ext::select_from(stride_A, idx_A, idx_ABC);
    auto stride_B_ABC = stl_ext::select_from(stride_B, idx_B, idx_ABC);
    auto stride_C_ABC = stl_ext::select_from(stride_C, idx_C, idx_ABC);

    auto idx_AB = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_B), idx_ABC);
    auto len_AB = stl_ext::select_from(len_A, idx_A, idx_AB);
    TBLIS_ASSERT(len_AB == stl_ext::select_from(len_B, idx_B, idx_AB));
    auto stride_A_AB = stl_ext::select_from(stride_A, idx_A, idx_AB);
    auto stride_B_AB = stl_ext::select_from(stride_B, idx_B, idx_AB);

    auto idx_AC = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_C), idx_ABC);
    auto len_AC = stl_ext::select_from(len_A, idx_A, idx_AC);
    TBLIS_ASSERT(len_AC == stl_ext::select_from(len_C, idx_C, idx_AC));
    auto stride_A_AC = stl_ext::select_from(stride_A, idx_A, idx_AC);
    auto stride_C_AC = stl_ext::select_from(stride_C, idx_C, idx_AC);

    auto idx_BC = stl_ext::exclusion(stl_ext::intersection(idx_B, idx_C), idx_ABC);
    auto len_BC = stl_ext::select_from(len_B, idx_B, idx_BC);
    TBLIS_ASSERT(len_BC == stl_ext::select_from(len_C, idx_C, idx_BC));
    auto stride_B_BC = stl_ext::select_from(stride_B, idx_B, idx_BC);
    auto stride_C_BC = stl_ext::select_from(stride_C, idx_C, idx_BC);

    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB, idx_AC, idx_ABC);
    auto len_A_only = stl_ext::select_from(len_A, idx_A, idx_A_only);
    auto stride_A_only = stl_ext::select_from(stride_A, idx_A, idx_A_only);
    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB, idx_BC, idx_ABC);
    auto len_B_only = stl_ext::select_from(len_B, idx_B, idx_B_only);
    auto stride_B_only = stl_ext::select_from(stride_B, idx_B, idx_B_only);
    auto idx_C_only = stl_ext::exclusion(idx_C, idx_AC, idx_BC, idx_ABC);
    auto len_C_only = stl_ext::select_from(len_C, idx_C, idx_C_only);
    auto stride_C_only = stl_ext::select_from(stride_C, idx_C, idx_C_only);

    TBLIS_ASSERT(stl_ext::intersection(idx_A_only, idx_B_only).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_A_only, idx_C_only).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_A_only, idx_AB).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_A_only, idx_AC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_A_only, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_A_only, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_B_only, idx_C_only).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_B_only, idx_AB).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_B_only, idx_AC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_B_only, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_B_only, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_C_only, idx_AB).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_C_only, idx_AC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_C_only, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_C_only, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_AC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AC, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AC, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_BC, idx_ABC).empty());

    fold(len_ABC, idx_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);
    fold(len_AB, idx_AB, stride_A_AB, stride_B_AB);
    fold(len_AC, idx_AC, stride_A_AC, stride_C_AC);
    fold(len_BC, idx_BC, stride_B_BC, stride_C_BC);
    fold(len_A_only, idx_A_only, stride_A_only);
    fold(len_B_only, idx_B_only, stride_B_only);
    fold(len_C_only, idx_C_only, stride_C_only);

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        T alpha = A->alpha<T>()*B->alpha<T>();
        T beta = C->alpha<T>();

        if (alpha == T(0))
        {
            if (beta == T(0))
            {
                parallelize_if(internal::set<T>, comm, get_config(cfg),
                               len_C_only+len_AC+len_BC+len_ABC,
                               T(0), static_cast<T*>(C->data),
                               stride_C_only+stride_C_AC+stride_C_BC+stride_C_ABC);
            }
            else
            {
                parallelize_if(internal::scale<T>, comm, get_config(cfg),
                               len_C_only+len_AC+len_BC+len_ABC,
                               beta, C->conj, static_cast<T*>(C->data),
                               stride_C_only+stride_C_AC+stride_C_BC+stride_C_ABC);
            }
        }
        else
        {
            parallelize_if(internal::mult<T>, comm, get_config(cfg),
                           len_A_only, len_B_only, len_C_only,
                           len_AB, len_AC, len_BC, len_ABC,
                           alpha, A->conj, static_cast<const T*>(A->data),
                           stride_A_only, stride_A_AB, stride_A_AC, stride_A_ABC,
                                  B->conj, static_cast<const T*>(B->data),
                           stride_B_only, stride_B_AB, stride_B_BC, stride_B_ABC,
                            beta, C->conj,       static_cast<T*>(C->data),
                           stride_C_only, stride_C_AC, stride_C_BC, stride_C_ABC);
        }

        C->alpha<T>() = T(1);
        C->conj = false;
    })
}

}

}
