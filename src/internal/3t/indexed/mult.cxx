#include "mult.hpp"
#include "internal/1t/indexed/util.hpp"
#include "internal/1t/dense/scale.hpp"
#include "internal/1t/dense/set.hpp"
#include "internal/3t/dense/mult.hpp"

#include "util/gemm_thread.hpp"
#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void mult_full(const communicator& comm, const config& cfg,
               T alpha, const indexed_varray_view<const T>& A,
               const dim_vector& idx_A_AB,
               const dim_vector& idx_A_AC,
               const dim_vector& idx_A_ABC,
                        const indexed_varray_view<const T>& B,
               const dim_vector& idx_B_AB,
               const dim_vector& idx_B_BC,
               const dim_vector& idx_B_ABC,
               T  beta, const indexed_varray_view<      T>& C,
               const dim_vector& idx_C_AC,
               const dim_vector& idx_C_BC,
               const dim_vector& idx_C_ABC)
{
    varray<T> A2, B2, C2;

    comm.broadcast(
    [&](varray<T>& A2, varray<T>& B2, varray<T>& C2)
    {
        if (comm.master())
        {
            block_to_full(A, A2);
            block_to_full(B, B2);
            block_to_full(C, C2);
        }

        auto len_AB = stl_ext::select_from(A2.lengths(), idx_A_AB);
        auto len_AC = stl_ext::select_from(C2.lengths(), idx_C_AC);
        auto len_BC = stl_ext::select_from(C2.lengths(), idx_C_BC);
        auto len_ABC = stl_ext::select_from(C2.lengths(), idx_C_ABC);
        auto stride_A_AB = stl_ext::select_from(A2.strides(), idx_A_AB);
        auto stride_A_AC = stl_ext::select_from(A2.strides(), idx_A_AC);
        auto stride_B_AB = stl_ext::select_from(B2.strides(), idx_B_AB);
        auto stride_B_BC = stl_ext::select_from(B2.strides(), idx_B_BC);
        auto stride_C_AC = stl_ext::select_from(C2.strides(), idx_C_AC);
        auto stride_C_BC = stl_ext::select_from(C2.strides(), idx_C_BC);
        auto stride_A_ABC = stl_ext::select_from(A2.strides(), idx_A_ABC);
        auto stride_B_ABC = stl_ext::select_from(B2.strides(), idx_B_ABC);
        auto stride_C_ABC = stl_ext::select_from(C2.strides(), idx_C_ABC);

        mult(comm, cfg, len_AB, len_AC, len_BC, len_ABC,
             alpha, false, A2.data(), stride_A_AB, stride_A_AC, stride_A_ABC,
                    false, B2.data(), stride_B_AB, stride_B_BC, stride_B_ABC,
              beta, false, C2.data(), stride_C_AC, stride_C_BC, stride_C_ABC);

        if (comm.master())
        {
            full_to_block(C2, C);
        }
    },
    A2, B2, C2);
}

template <typename T>
void contract_block(const communicator& comm, const config& cfg,
                    T alpha, indexed_varray_view<const T> A,
                    dim_vector idx_A_AB,
                    dim_vector idx_A_AC,
                             indexed_varray_view<const T> B,
                    dim_vector idx_B_AB,
                    dim_vector idx_B_BC,
                    T  beta, indexed_varray_view<      T> C,
                    dim_vector idx_C_AC,
                    dim_vector idx_C_BC)
{
    index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    index_group<2> group_AC(A, idx_A_AC, C, idx_C_AC);
    index_group<2> group_BC(B, idx_B_BC, C, idx_C_BC);

    group_indices<2> indices_A(A, group_AC, 0, group_AB, 0);
    group_indices<2> indices_B(B, group_BC, 0, group_AB, 1);
    group_indices<2> indices_C(C, group_AC, 1, group_BC, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();
    auto nidx_C = indices_C.size();

    dynamic_task_set tasks(comm, nidx_C, stl_ext::prod(group_AB.dense_len)*
                                         stl_ext::prod(group_AC.dense_len)*
                                         stl_ext::prod(group_BC.dense_len));

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_C = 0;

    while (idx_A < nidx_A && idx_C < nidx_C)
    {
        if (indices_A[idx_A].key[0] < indices_C[idx_C].key[0])
        {
            idx_A++;
        }
        else if (indices_A[idx_A].key[0] > indices_C[idx_C].key[0])
        {
            if (beta != T(1))
            {
                tasks.visit(idx++,
                [&,idx_C](const communicator& subcomm)
                {
                    auto data_C = C.data(0) + indices_C[idx_C].offset[0];

                    if (beta == T(0))
                    {
                        set(subcomm, cfg, C.dense_lengths(),
                            T(0), data_C, C.dense_strides());
                    }
                    else
                    {
                        scale(subcomm, cfg, C.dense_lengths(),
                              beta, false, data_C, C.dense_strides());
                    }
                });
            }

            idx_C++;
        }
        else
        {
            auto next_A = idx_A;

            do { next_A++; }
            while (next_A < nidx_A &&
                   indices_A[idx_A].key[0] == indices_C[idx_C].key[0]);

            stride_type idx_B = 0;

            while (idx_B < nidx_B && idx_C < nidx_C &&
                   indices_A[idx_A].key[0] == indices_C[idx_C].key[0])
            {
                if (indices_B[idx_B].key[0] < indices_C[idx_C].key[1])
                {
                    idx_B++;
                }
                else if (indices_B[idx_B].key[0] > indices_C[idx_C].key[1])
                {
                    if (beta != T(1))
                    {
                        tasks.visit(idx++,
                        [&,idx_C](const communicator& subcomm)
                        {
                            auto data_C = C.data(0) + indices_C[idx_C].offset[0];

                            if (beta == T(0))
                            {
                                set(subcomm, cfg, C.dense_lengths(),
                                    T(0), data_C, C.dense_strides());
                            }
                            else
                            {
                                scale(subcomm, cfg, C.dense_lengths(),
                                      beta, false, data_C, C.dense_strides());
                            }
                        });
                    }

                    idx_C++;
                }
                else
                {
                    auto next_B = idx_B;

                    do { next_B++; }
                    while (next_B < nidx_B &&
                           indices_B[idx_B].key[0] == indices_C[idx_C].key[1]);

                    tasks.visit(idx++,
                    [&,idx_A,idx_B,idx_C,next_A,next_B,beta]
                    (const communicator& subcomm)
                    {
                        stride_type off_A_AC, off_C_AC;
                        get_local_offset(indices_A[idx_A].idx[0], group_AC,
                                         off_A_AC, 0, off_C_AC, 1);

                        stride_type off_B_BC, off_C_BC;
                        get_local_offset(indices_B[idx_B].idx[0], group_BC,
                                         off_B_BC, 0, off_C_BC, 1);

                        auto data_C = C.data(0) + indices_C[idx_C].offset;

                        if (!group_AC.mixed_pos[1].empty() ||
                            !group_BC.mixed_pos[1].empty())
                        {
                            if (beta == T(0))
                            {
                                set(comm, cfg, C.dense_lengths(),
                                    beta, data_C, C.dense_strides());
                            }
                            else if (beta != T(1))
                            {
                                scale(comm, cfg, C.dense_lengths(),
                                      beta, false, data_C, C.dense_strides());
                            }

                            beta = T(1);
                        }

                        data_C += off_C_AC + off_C_BC;

                        while (idx_A < next_A && idx_B < next_B)
                        {
                            if (indices_A[idx_A].key[1] < indices_B[idx_B].key[1])
                            {
                                idx_A++;
                            }
                            else if (indices_A[idx_A].key[1] > indices_B[idx_B].key[1])
                            {
                                idx_B++;
                            }
                            else
                            {
                                stride_type off_A_AB, off_B_AB;
                                get_local_offset(indices_A[idx_A].idx[0], group_AB,
                                                 off_A_AB, 0, off_B_AB, 1);

                                auto data_A = A.data(0) + indices_A[idx_A].offset + off_A_AB + off_A_AC;
                                auto data_B = B.data(0) + indices_B[idx_B].offset + off_B_AB + off_B_BC;

                                mult(subcomm, cfg,
                                     group_AC.dense_len,
                                     group_BC.dense_len,
                                     group_AB.dense_len, {},
                                     alpha, false, data_A, group_AC.dense_stride[0],
                                                           group_AB.dense_stride[0], {},
                                            false, data_B, group_BC.dense_stride[0],
                                                           group_AB.dense_stride[1], {},
                                      beta, false, data_C, group_AC.dense_stride[1],
                                                           group_BC.dense_stride[1], {});

                                beta = T(1);
                            }
                        }

                        data_C -= off_C_AC + off_C_BC;

                        if (beta == T(0))
                        {
                            set(comm, cfg, C.dense_lengths(),
                                beta, data_C, C.dense_strides());
                        }
                        else if (beta != T(1))
                        {
                            scale(comm, cfg, C.dense_lengths(),
                                  beta, false, data_C, C.dense_strides());
                        }
                    });

                    idx_B = next_B;
                    idx_C++;
                }
            }

            idx_A = next_A;
            idx_C++;
        }
    }

    while (idx_C < nidx_C)
    {
        tasks.visit(idx++,
        [&,idx_C]
        (const communicator& subcomm)
        {
            auto data_C = C.data(0) + indices_C[idx_C].offset;

            if (beta == T(0))
            {
                set(comm, cfg, C.dense_lengths(),
                    beta, data_C, C.dense_strides());
            }
            else if (beta != T(1))
            {
                scale(comm, cfg, C.dense_lengths(),
                      beta, false, data_C, C.dense_strides());
            }
        });

        idx_C++;
    }
}

template <typename T>
void mult_block(const communicator& comm, const config& cfg,
                T alpha, indexed_varray_view<const T> A,
                dim_vector idx_A_AB,
                dim_vector idx_A_AC,
                dim_vector idx_A_ABC,
                         indexed_varray_view<const T> B,
                dim_vector idx_B_AB,
                dim_vector idx_B_BC,
                dim_vector idx_B_ABC,
                T  beta, indexed_varray_view<      T> C,
                dim_vector idx_C_AC,
                dim_vector idx_C_BC,
                dim_vector idx_C_ABC)
{
    index_group<3> group_ABC(A, idx_A_ABC, B, idx_B_ABC, C, idx_C_ABC);
    index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    index_group<2> group_AC(A, idx_A_AC, C, idx_C_AC);
    index_group<2> group_BC(B, idx_B_BC, C, idx_C_BC);

    group_indices<3> indices_A(A, group_ABC, 0, group_AC, 0, group_AB, 0);
    group_indices<3> indices_B(B, group_ABC, 1, group_BC, 0, group_AB, 1);
    group_indices<3> indices_C(C, group_ABC, 2, group_AC, 1, group_BC, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();
    auto nidx_C = indices_C.size();

    dynamic_task_set tasks(comm, nidx_C, stl_ext::prod(group_AB.dense_len)*
                                         stl_ext::prod(group_AC.dense_len)*
                                         stl_ext::prod(group_BC.dense_len)*
                                         stl_ext::prod(group_ABC.dense_len));

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_B0 = 0;
    stride_type idx_C = 0;

    while (idx_A < nidx_A && idx_B0 < nidx_B && idx_C < nidx_C)
    {
        if (indices_A[idx_A].key[0] < indices_B[idx_B0].key[0])
        {
            if (indices_A[idx_A].key[0] < indices_C[idx_C].key[0])
            {
                idx_A++;
            }
            else
            {
                if (beta != T(1))
                {
                    tasks.visit(idx++,
                    [&,idx_C](const communicator& subcomm)
                    {
                        auto data_C = C.data(0) + indices_C[idx_C].offset[0];

                        if (beta == T(0))
                        {
                            set(subcomm, cfg, C.dense_lengths(),
                                T(0), data_C, C.dense_strides());
                        }
                        else
                        {
                            scale(subcomm, cfg, C.dense_lengths(),
                                  beta, false, data_C, C.dense_strides());
                        }
                    });
                }

                idx_C++;
            }
        }
        else if (indices_A[idx_A].key[0] > indices_B[idx_B0].key[0])
        {
            if (indices_B[idx_B0].key[0] < indices_C[idx_C].key[0])
            {
                idx_B0++;
            }
            else
            {
                if (beta != T(1))
                {
                    tasks.visit(idx++,
                    [&,idx_C](const communicator& subcomm)
                    {
                        auto data_C = C.data(0) + indices_C[idx_C].offset[0];

                        if (beta == T(0))
                        {
                            set(subcomm, cfg, C.dense_lengths(),
                                T(0), data_C, C.dense_strides());
                        }
                        else
                        {
                            scale(subcomm, cfg, C.dense_lengths(),
                                  beta, false, data_C, C.dense_strides());
                        }
                    });
                }

                idx_C++;
            }
        }
        else
        {
            if (indices_A[idx_A].key[0] < indices_C[idx_C].key[0])
            {
                idx_A++;
                idx_B0++;
            }
            else if (indices_A[idx_A].key[0] > indices_C[idx_C].key[0])
            {
                if (beta != T(1))
                {
                    tasks.visit(idx++,
                    [&,idx_C](const communicator& subcomm)
                    {
                        auto data_C = C.data(0) + indices_C[idx_C].offset[0];

                        if (beta == T(0))
                        {
                            set(subcomm, cfg, C.dense_lengths(),
                                T(0), data_C, C.dense_strides());
                        }
                        else
                        {
                            scale(subcomm, cfg, C.dense_lengths(),
                                  beta, false, data_C, C.dense_strides());
                        }
                    });
                }

                idx_C++;
            }
            else
            {
                auto next_A_ABC = idx_A;
                auto next_B_ABC = idx_B0;
                auto next_C_ABC = idx_C;

                do { next_A_ABC++; }
                while (next_A_ABC < nidx_A &&
                       indices_A[next_A_ABC].key[0] == indices_A[idx_A].key[0]);

                do { next_B_ABC++; }
                while (next_B_ABC < nidx_B &&
                       indices_B[next_B_ABC].key[0] == indices_B[idx_B0].key[0]);

                do { next_C_ABC++; }
                while (next_C_ABC < nidx_C &&
                       indices_C[next_C_ABC].key[0] == indices_C[idx_C].key[0]);

                while (idx_A < next_A_ABC && idx_C < next_C_ABC)
                {
                    if (indices_A[idx_A].key[1] < indices_C[idx_C].key[1])
                    {
                        idx_A++;
                    }
                    else if (indices_A[idx_A].key[1] > indices_C[idx_C].key[1])
                    {
                        if (beta != T(1))
                        {
                            tasks.visit(idx++,
                            [&,idx_C](const communicator& subcomm)
                            {
                                auto data_C = C.data(0) + indices_C[idx_C].offset[0];

                                if (beta == T(0))
                                {
                                    set(subcomm, cfg, C.dense_lengths(),
                                        T(0), data_C, C.dense_strides());
                                }
                                else
                                {
                                    scale(subcomm, cfg, C.dense_lengths(),
                                          beta, false, data_C, C.dense_strides());
                                }
                            });
                        }

                        idx_C++;
                    }
                    else
                    {
                        auto next_A_AB = idx_A;

                        do { next_A_AB++; }
                        while (next_A_AB < next_A_ABC &&
                               indices_A[next_A_AB].key[1] == indices_A[idx_A].key[1]);

                        stride_type idx_B = idx_B0;

                        while (idx_B < next_B_ABC && idx_C < next_C_ABC &&
                               indices_A[idx_A].key[1] == indices_C[idx_C].key[1])
                        {
                            if (indices_B[idx_B].key[1] < indices_C[idx_C].key[2])
                            {
                                idx_B++;
                            }
                            else if (indices_B[idx_B].key[1] > indices_C[idx_C].key[2])
                            {
                                if (beta != T(1))
                                {
                                    tasks.visit(idx++,
                                    [&,idx_C](const communicator& subcomm)
                                    {
                                        auto data_C = C.data(0) + indices_C[idx_C].offset[0];

                                        if (beta == T(0))
                                        {
                                            set(subcomm, cfg, C.dense_lengths(),
                                                T(0), data_C, C.dense_strides());
                                        }
                                        else
                                        {
                                            scale(subcomm, cfg, C.dense_lengths(),
                                                  beta, false, data_C, C.dense_strides());
                                        }
                                    });
                                }

                                idx_C++;
                            }
                            else
                            {
                                auto next_B_AB = idx_B;

                                do { next_B_AB++; }
                                while (next_B_AB < next_B_ABC &&
                                       indices_B[next_B_AB].key[1] == indices_B[idx_B].key[1]);

                                tasks.visit(idx++,
                                [&,idx_A,idx_B,idx_C,next_A_AB,next_B_AB,beta]
                                (const communicator& subcomm)
                                {
                                    stride_type off_A_ABC, off_B_ABC, off_C_ABC;
                                    get_local_offset(indices_A[idx_A].idx[0], group_ABC,
                                                     off_A_ABC, 0, off_B_ABC, 1, off_C_ABC, 2);

                                    stride_type off_A_AC, off_C_AC;
                                    get_local_offset(indices_A[idx_A].idx[1], group_AC,
                                                     off_A_AC, 0, off_C_AC, 1);

                                    stride_type off_B_BC, off_C_BC;
                                    get_local_offset(indices_B[idx_B].idx[1], group_BC,
                                                     off_B_BC, 0, off_C_BC, 1);

                                    auto data_C = C.data(0) + indices_C[idx_C].offset;

                                    if (!group_AC.mixed_pos[1].empty() ||
                                        !group_BC.mixed_pos[1].empty())
                                    {
                                        if (beta == T(0))
                                        {
                                            set(comm, cfg, C.dense_lengths(),
                                                beta, data_C, C.dense_strides());
                                        }
                                        else if (beta != T(1))
                                        {
                                            scale(comm, cfg, C.dense_lengths(),
                                                  beta, false, data_C, C.dense_strides());
                                        }

                                        beta = T(1);
                                    }

                                    data_C += off_C_AC + off_C_BC + off_C_ABC;

                                    while (idx_A < next_A_AB && idx_B < next_B_AB)
                                    {
                                        if (indices_A[idx_A].key[2] < indices_B[idx_B].key[2])
                                        {
                                            idx_A++;
                                        }
                                        else if (indices_A[idx_A].key[2] > indices_B[idx_B].key[2])
                                        {
                                            idx_B++;
                                        }
                                        else
                                        {
                                            stride_type off_A_AB, off_B_AB;
                                            get_local_offset(indices_A[idx_A].idx[1], group_AB,
                                                             off_A_AB, 0, off_B_AB, 1);

                                            auto data_A = A.data(0) + indices_A[idx_A].offset + off_A_AB + off_A_AC + off_A_ABC;
                                            auto data_B = B.data(0) + indices_B[idx_B].offset + off_B_AB + off_B_BC + off_B_ABC;

                                            mult(subcomm, cfg,
                                                 group_AC.dense_len,
                                                 group_BC.dense_len,
                                                 group_AB.dense_len,
                                                 group_ABC.dense_len,
                                                 alpha, false, data_A, group_AC.dense_stride[0],
                                                                       group_AB.dense_stride[0],
                                                                       group_ABC.dense_stride[0],
                                                        false, data_B, group_BC.dense_stride[0],
                                                                       group_AB.dense_stride[1],
                                                                       group_ABC.dense_stride[1],
                                                  beta, false, data_C, group_AC.dense_stride[1],
                                                                       group_BC.dense_stride[1],
                                                                       group_ABC.dense_stride[2]);

                                            beta = T(1);
                                        }
                                    }

                                    data_C -= off_C_AC + off_C_BC + off_C_ABC;

                                    if (beta == T(0))
                                    {
                                        set(comm, cfg, C.dense_lengths(),
                                            beta, data_C, C.dense_strides());
                                    }
                                    else if (beta != T(1))
                                    {
                                        scale(comm, cfg, C.dense_lengths(),
                                              beta, false, data_C, C.dense_strides());
                                    }
                                });

                                idx_B = next_B_AB;
                                idx_C++;
                            }
                        }

                        idx_A = next_A_AB;
                        idx_C++;
                    }
                }

                idx_A = next_A_ABC;
                idx_B0 = next_B_ABC;
                idx_C = next_C_ABC;
            }
        }
    }

    while (idx_C < nidx_C)
    {
        tasks.visit(idx++,
        [&,idx_C]
        (const communicator& subcomm)
        {
            auto data_C = C.data(0) + indices_C[idx_C].offset;

            if (beta == T(0))
            {
                set(comm, cfg, C.dense_lengths(),
                    beta, data_C, C.dense_strides());
            }
            else if (beta != T(1))
            {
                scale(comm, cfg, C.dense_lengths(),
                      beta, false, data_C, C.dense_strides());
            }
        });

        idx_C++;
    }
}

template <typename T>
void mult(const communicator& comm, const config& cfg,
          T alpha, bool conj_A, const indexed_varray_view<const T>& A,
          const dim_vector& idx_A_AB,
          const dim_vector& idx_A_AC,
          const dim_vector& idx_A_ABC,
                   bool conj_B, const indexed_varray_view<const T>& B,
          const dim_vector& idx_B_AB,
          const dim_vector& idx_B_BC,
          const dim_vector& idx_B_ABC,
          T  beta, bool conj_C, const indexed_varray_view<      T>& C,
          const dim_vector& idx_C_AC,
          const dim_vector& idx_C_BC,
          const dim_vector& idx_C_ABC)
{
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C);

    if (dpd_impl == FULL)
    {
        mult_full(comm, cfg,
                  alpha, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                         B, idx_B_AB, idx_B_BC, idx_B_ABC,
                   beta, C, idx_C_AC, idx_C_BC, idx_C_ABC);
    }
    else if (idx_C_ABC.empty())
    {
        contract_block(comm, cfg,
                       alpha, A, idx_A_AB, idx_A_AC,
                              B, idx_B_AB, idx_B_BC,
                        beta, C, idx_C_AC, idx_C_BC);
    }
    else
    {
        mult_block(comm, cfg,
                   alpha, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                          B, idx_B_AB, idx_B_BC, idx_B_ABC,
                    beta, C, idx_C_AC, idx_C_BC, idx_C_ABC);
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, const config& cfg, \
                   T alpha, bool conj_A, const indexed_varray_view<const T>& A, \
                   const dim_vector& idx_A_AB, \
                   const dim_vector& idx_A_AC, \
                   const dim_vector& idx_A_ABC, \
                            bool conj_B, const indexed_varray_view<const T>& B, \
                   const dim_vector& idx_B_AB, \
                   const dim_vector& idx_B_BC, \
                   const dim_vector& idx_B_ABC, \
                   T  beta, bool conj_C, const indexed_varray_view<      T>& C, \
                   const dim_vector& idx_C_AC, \
                   const dim_vector& idx_C_BC, \
                   const dim_vector& idx_C_ABC);
#include "configs/foreach_type.h"

}
}
