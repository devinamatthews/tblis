#include "mult.hpp"

#include "util/gemm_thread.hpp"
#include "util/tensor.hpp"

#include "nodes/gemm.hpp"

#include "internal/1t/dense/add.hpp"
#include "internal/1t/dense/dot.hpp"
#include "internal/3m/mult.hpp"

namespace tblis
{

MemoryPool BuffersForScatter(4096);

namespace internal
{

impl_t impl = BLIS_BASED;

template <typename T>
void contract_blas(const communicator& comm, const config& cfg,
                   const len_vector& len_AB,
                   const len_vector& len_AC,
                   const len_vector& len_BC,
                   T alpha, bool conj_A, const T* A,
                   const stride_vector& stride_A_AB,
                   const stride_vector& stride_A_AC,
                            bool conj_B, const T* B,
                   const stride_vector& stride_B_AB,
                   const stride_vector& stride_B_BC,
                   T  beta, bool conj_C,       T* C,
                   const stride_vector& stride_C_AC,
                   const stride_vector& stride_C_BC)
{
    varray<T> ar, br, cr;

    if (comm.master())
    {
        ar.reset(len_AC+len_AB);
        br.reset(len_AB+len_BC);
        cr.reset(len_AC+len_BC);
    }

    comm.broadcast(
    [&](varray<T>& ar, varray<T>& br, varray<T>& cr)
    {
        matrix_view<T> am, bm, cm;
        matricize<T>(ar, am, static_cast<unsigned>(len_AC.size()));
        matricize<T>(br, bm, static_cast<unsigned>(len_AB.size()));
        matricize<T>(cr, cm, static_cast<unsigned>(len_AC.size()));

        add(comm, cfg, {}, {}, ar.lengths(),
            T(1), conj_A,         A, {}, stride_A_AC+stride_A_AB,
            T(0),  false, ar.data(), {},            ar.strides());

        add(comm, cfg, {}, {}, br.lengths(),
            T(1), conj_B,         B, {}, stride_B_AB+stride_B_BC,
            T(0),  false, br.data(), {},            br.strides());

        mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
             alpha, false, am.data(), am.stride(0), am.stride(1),
                    false, bm.data(), bm.stride(0), bm.stride(1),
              T(0), false, cm.data(), cm.stride(0), cm.stride(1));

        add(comm, cfg, {}, {}, cr.lengths(),
            T(1),  false, cr.data(), {},            cr.strides(),
            beta, conj_C,         C, {}, stride_C_AC+stride_C_BC);
    },
    ar, br, cr);
}

template <typename T>
void contract_blis(const communicator& comm, const config& cfg,
                   const len_vector& len_AB,
                   const len_vector& len_AC,
                   const len_vector& len_BC,
                   T alpha, bool conj_A, const T* A,
                   const stride_vector& stride_A_AB,
                   const stride_vector& stride_A_AC,
                            bool conj_B, const T* B,
                   const stride_vector& stride_B_AB,
                   const stride_vector& stride_B_BC,
                   T  beta, bool conj_C,       T* C,
                   const stride_vector& stride_C_AC,
                   const stride_vector& stride_C_BC)
{
    //TODO
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C);

    auto reorder_AC = detail::sort_by_stride(stride_C_AC, stride_A_AC);
    auto reorder_BC = detail::sort_by_stride(stride_C_BC, stride_B_BC);
    auto reorder_AB = detail::sort_by_stride(stride_A_AB, stride_B_AB);

    tensor_matrix<T> at(stl_ext::permuted(len_AC, reorder_AC),
                        stl_ext::permuted(len_AB, reorder_AB),
                        const_cast<T*>(A),
                        stl_ext::permuted(stride_A_AC, reorder_AC),
                        stl_ext::permuted(stride_A_AB, reorder_AB));

    tensor_matrix<T> bt(stl_ext::permuted(len_AB, reorder_AB),
                        stl_ext::permuted(len_BC, reorder_BC),
                        const_cast<T*>(B),
                        stl_ext::permuted(stride_B_AB, reorder_AB),
                        stl_ext::permuted(stride_B_BC, reorder_BC));

    tensor_matrix<T> ct(stl_ext::permuted(len_AC, reorder_AC),
                        stl_ext::permuted(len_BC, reorder_BC),
                        C,
                        stl_ext::permuted(stride_C_AC, reorder_AC),
                        stl_ext::permuted(stride_C_BC, reorder_BC));

    const bool row_major = cfg.gemm_row_major.value<T>();

    if (ct.stride(!row_major) == 1)
    {
        /*
         * Compute C^T = B^T * A^T instead
         */
        at.swap(bt);
        at.transpose();
        bt.transpose();
        ct.transpose();
    }

    TensorGEMM gemm;

    len_type m = ct.length(0);
    len_type n = ct.length(1);
    len_type k = at.length(1);

    int nt = max_num_threads(comm);
    auto tc = make_gemm_thread_config<T>(cfg, nt, m, n, k);
    step<0>(gemm).distribute = tc.jc_nt;
    step<4>(gemm).distribute = tc.ic_nt;
    step<8>(gemm).distribute = tc.jr_nt;
    step<9>(gemm).distribute = tc.ir_nt;

    gemm(comm, cfg, alpha, at, bt, beta, ct);
}

#define FOREACH_TYPE(T) \
template void contract_blis(const communicator& comm, const config& cfg, \
                            const len_vector& len_AB, \
                            const len_vector& len_AC, \
                            const len_vector& len_BC, \
                            T alpha, bool conj_A, const T* A, \
                            const stride_vector& stride_A_AB, \
                            const stride_vector& stride_A_AC, \
                                     bool conj_B, const T* B, \
                            const stride_vector& stride_B_AB, \
                            const stride_vector& stride_B_BC, \
                            T  beta, bool conj_C,       T* C, \
                            const stride_vector& stride_C_AC, \
                            const stride_vector& stride_C_BC);
#include "configs/foreach_type.h"

template <typename T>
void mult_blas(const communicator& comm, const config& cfg,
               const len_vector& len_AB,
               const len_vector& len_AC,
               const len_vector& len_BC,
               const len_vector& len_ABC,
               T alpha, bool conj_A, const T* A,
               const stride_vector& stride_A_AB,
               const stride_vector& stride_A_AC,
               const stride_vector& stride_A_ABC,
                        bool conj_B, const T* B,
               const stride_vector& stride_B_AB,
               const stride_vector& stride_B_BC,
               const stride_vector& stride_B_ABC,
               T  beta, bool conj_C,       T* C,
               const stride_vector& stride_C_AC,
               const stride_vector& stride_C_BC,
               const stride_vector& stride_C_ABC)
{
    varray<T> ar, br, cr;

    if (comm.master())
    {
        ar.reset(len_AC+len_AB);
        br.reset(len_AB+len_BC);
        cr.reset(len_AC+len_BC);
    }

    comm.broadcast(
    [&](varray<T>& ar, varray<T>& br, varray<T>& cr)
    {
        matrix_view<T> am, bm, cm;
        matricize<T>(ar, am, static_cast<unsigned>(len_AC.size()));
        matricize<T>(br, bm, static_cast<unsigned>(len_AB.size()));
        matricize<T>(cr, cm, static_cast<unsigned>(len_AC.size()));

        viterator<3> it(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

        while (it.next(A, B, C))
        {
            add(comm, cfg, {}, {}, ar.lengths(),
                T(1), conj_A,         A, {}, stride_A_AC+stride_A_AB,
                T(0),  false, ar.data(), {},            ar.strides());

            add(comm, cfg, {}, {}, br.lengths(),
                T(1), conj_B,         B, {}, stride_B_AB+stride_B_BC,
                T(0),  false, br.data(), {},            br.strides());

            mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
                 alpha, false, am.data(), am.stride(0), am.stride(1),
                        false, bm.data(), bm.stride(0), bm.stride(1),
                  T(0), false, cm.data(), cm.stride(0), cm.stride(1));

            add(comm, cfg, {}, {}, cr.lengths(),
                T(1),  false, cr.data(), {},             cr.strides(),
                beta, conj_C,         C, {}, stride_C_AC+stride_C_BC);
        }
    },
    ar, br, cr);
}

template <typename T>
void mult_ref(const communicator& comm, const config& cfg,
              const len_vector& len_AB,
              const len_vector& len_AC,
              const len_vector& len_BC,
              const len_vector& len_ABC,
              T alpha, bool conj_A, const T* A,
              const stride_vector& stride_A_AB,
              const stride_vector& stride_A_AC,
              const stride_vector& stride_A_ABC,
                       bool conj_B, const T* B,
              const stride_vector& stride_B_AB,
              const stride_vector& stride_B_BC,
              const stride_vector& stride_B_ABC,
              T  beta, bool conj_C,       T* C,
              const stride_vector& stride_C_AC,
              const stride_vector& stride_C_BC,
              const stride_vector& stride_C_ABC)
{
    (void)cfg;

    viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
    viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);
    viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);
    len_type n = stl_ext::prod(len_ABC);

    len_type n_min, n_max;
    std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(n);

    iter_ABC.position(n_min, A, B, C);

    for (len_type i = n_min;i < n_max;i++)
    {
        iter_ABC.next(A, B, C);

        while (iter_AC.next(A, C))
        {
            while (iter_BC.next(B, C))
            {
                T temp = T();

                TBLIS_SPECIAL_CASE(conj_A,
                TBLIS_SPECIAL_CASE(conj_B,
                while (iter_AB.next(A, B))
                {
                    temp += (conj_A ? conj(*A) : *A)*
                            (conj_B ? conj(*B) : *B);
                }
                ))
                temp *= alpha;

                if (beta == T(0))
                {
                    *C = temp;
                }
                else
                {
                    *C = temp + beta*(conj_C ? conj(*C) : *C);
                }
            }
        }
    }
}

template <typename T>
void mult_vec(const communicator& comm, const config& cfg,
              const len_vector& len_ABC,
              T alpha, bool conj_A, const T* A,
              const stride_vector& stride_A_ABC,
                       bool conj_B, const T* B,
              const stride_vector& stride_B_ABC,
              T  beta, bool conj_C,       T* C,
              const stride_vector& stride_C_ABC)
{
    (void)cfg;

    viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);
    len_type n = stl_ext::prod(len_ABC);

    len_type n_min, n_max;
    std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(n);

    iter_ABC.position(n_min, A, B, C);

    if (beta == T(0))
    {
        TBLIS_SPECIAL_CASE(conj_A,
        TBLIS_SPECIAL_CASE(conj_B,
        for (len_type i = n_min;i < n_max;i++)
        {
            iter_ABC.next(A, B, C);

            *C = alpha*(conj_A ? conj(*A) : *A)*
                       (conj_B ? conj(*B) : *B);
        }
        ))
    }
    else
    {
        TBLIS_SPECIAL_CASE(conj_A,
        TBLIS_SPECIAL_CASE(conj_B,
        for (len_type i = n_min;i < n_max;i++)
        {
            iter_ABC.next(A, B, C);

            *C = alpha*(conj_A ? conj(*A) : *A)*
                       (conj_B ? conj(*B) : *B) +
                  beta*(conj_C ? conj(*C) : *C);
        }
        ))
    }
}

template <typename T>
void outer_prod_blas(const communicator& comm, const config& cfg,
                     const len_vector& len_AC,
                     const len_vector& len_BC,
                     T alpha, bool conj_A, const T* A,
                     const stride_vector& stride_A_AC,
                              bool conj_B, const T* B,
                     const stride_vector& stride_B_BC,
                     T  beta, bool conj_C,       T* C,
                     const stride_vector& stride_C_AC,
                     const stride_vector& stride_C_BC)
{
    varray<T> ar, br, cr;

    if (comm.master())
    {
        ar.reset(len_AC);
        br.reset(len_BC);
        cr.reset(len_AC+len_BC);
    }

    comm.broadcast(
    [&](varray<T>& ar, varray<T>& br, varray<T>& cr)
    {
        matrix_view<T> am, bm, cm;
        matricize<T>(ar, am, static_cast<unsigned>(len_AC.size()));
        matricize<T>(br, bm, 0);
        matricize<T>(cr, cm, static_cast<unsigned>(len_AC.size()));

        add(comm, cfg, {}, {}, ar.lengths(),
            T(1), conj_A,         A, {},  stride_A_AC,
            T(0),  false, ar.data(), {}, ar.strides());

        add(comm, cfg, {}, {}, br.lengths(),
            T(1), conj_B,         B, {},  stride_B_BC,
            T(0),  false, br.data(), {}, br.strides());

        mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
             alpha, false, am.data(), am.stride(0), am.stride(1),
                    false, bm.data(), bm.stride(0), bm.stride(1),
              T(0), false, cm.data(), cm.stride(0), cm.stride(1));

        add(comm, cfg, {}, {}, cr.lengths(),
            T(1),  false, cr.data(), {},            cr.strides(),
            beta, conj_C,         C, {}, stride_C_AC+stride_C_BC);
    },
    ar, br, cr);
}

template <typename T>
void weight_blas(const communicator& comm, const config& cfg,
                 const len_vector& len_AC,
                 const len_vector& len_BC,
                 const len_vector& len_ABC,
                 T alpha, bool conj_A, const T* A,
                 const stride_vector& stride_A_AC,
                 const stride_vector& stride_A_ABC,
                          bool conj_B, const T* B,
                 const stride_vector& stride_B_BC,
                 const stride_vector& stride_B_ABC,
                 T  beta, bool conj_C,       T* C,
                 const stride_vector& stride_C_AC,
                 const stride_vector& stride_C_BC,
                 const stride_vector& stride_C_ABC)
{
    varray<T> ar, br, cr;

    if (comm.master())
    {
        ar.reset(len_AC);
        br.reset(len_BC);
        cr.reset(len_AC+len_BC);
    }

    comm.broadcast(
    [&](varray<T>& ar, varray<T>& br, varray<T>& cr)
    {
        matrix_view<T> am, bm, cm;
        matricize<T>(ar, am, static_cast<unsigned>(len_AC.size()));
        matricize<T>(br, bm, 0);
        matricize<T>(cr, cm, static_cast<unsigned>(len_AC.size()));

        viterator<3> it(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

        while (it.next(A, B, C))
        {
            add(comm, cfg, {}, {}, ar.lengths(),
                T(1), conj_A,         A, {},  stride_A_AC,
                T(0),  false, ar.data(), {}, ar.strides());

            add(comm, cfg, {}, {}, br.lengths(),
                T(1), conj_B,         B, {},  stride_B_BC,
                T(0),  false, br.data(), {}, br.strides());

            mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
                 alpha, false, am.data(), am.stride(0), am.stride(1),
                        false, bm.data(), bm.stride(0), bm.stride(1),
                  T(0), false, cm.data(), cm.stride(0), cm.stride(1));

            add(comm, cfg, {}, {}, cr.lengths(),
                T(1),  false, cr.data(), {},            cr.strides(),
                beta, conj_C,         C, {}, stride_C_AC+stride_C_BC);
        }
    },
    ar, br, cr);
}

template <typename T>
void mult(const communicator& comm, const config& cfg,
          const len_vector& len_AB,
          const len_vector& len_AC,
          const len_vector& len_BC,
          const len_vector& len_ABC,
          T alpha, bool conj_A, const T* A,
          const stride_vector& stride_A_AB,
          const stride_vector& stride_A_AC,
          const stride_vector& stride_A_ABC,
                   bool conj_B, const T* B,
          const stride_vector& stride_B_AB,
          const stride_vector& stride_B_BC,
          const stride_vector& stride_B_ABC,
          T  beta, bool conj_C,       T* C,
          const stride_vector& stride_C_AC,
          const stride_vector& stride_C_BC,
          const stride_vector& stride_C_ABC)
{
    if (impl == REFERENCE)
    {
        mult_ref(comm, cfg,
                 len_AB, len_AC, len_BC, len_ABC,
                 alpha, conj_A, A, stride_A_AB, stride_A_AC, stride_A_ABC,
                        conj_B, B, stride_B_AB, stride_B_BC, stride_B_ABC,
                  beta, conj_C, C, stride_C_AC, stride_C_BC, stride_C_ABC);
        return;
    }

    enum
    {
        HAS_NONE = 0x0,
        HAS_AB   = 0x1,
        HAS_AC   = 0x2,
        HAS_BC   = 0x4,
        HAS_ABC  = 0x8
    };

    int groups = (len_AB.empty()  ? 0 : HAS_AB ) +
                 (len_AC.empty()  ? 0 : HAS_AC ) +
                 (len_BC.empty()  ? 0 : HAS_BC ) +
                 (len_ABC.empty() ? 0 : HAS_ABC);

    T sum;
    viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

    switch (groups)
    {
        case HAS_NONE:
            if (beta == T(0))
            {
                *C = alpha*(conj_A ? conj(*A) : *A)*
                           (conj_B ? conj(*B) : *B);
            }
            else
            {
                *C = alpha*(conj_A ? conj(*A) : *A)*
                           (conj_B ? conj(*B) : *B) +
                      beta*(conj_C ? conj(*C) : *C);
            }
            break;
        case HAS_AB:
            dot(comm, cfg, len_AB, conj_A, A, stride_A_AB,
                                   conj_B, B, stride_B_AB, sum);
            if (beta == T(0))
            {
                *C = alpha*sum;
            }
            else
            {
                *C = alpha*sum + beta*(conj_C ? conj(*C) : *C);
            }
            break;
        case HAS_AC:
            add(comm, cfg, {}, {}, len_AC, alpha*(conj_B ? conj(*B) : *B),
                conj_A, A, {}, stride_A_AC, beta, conj_C, C, {}, stride_C_AC);
            break;
        case HAS_BC:
            add(comm, cfg, {}, {}, len_BC, alpha*(conj_A ? conj(*A) : *A),
                conj_B, B, {}, stride_B_BC, beta, conj_C, C, {}, stride_C_BC);
            break;
        case HAS_ABC:
            mult_vec(comm, cfg, len_ABC,
                     alpha, conj_A, A, stride_A_ABC,
                            conj_B, B, stride_B_ABC,
                      beta, conj_C, C, stride_C_ABC);
            break;
        case HAS_AC+HAS_BC:
            outer_prod_blas(comm, cfg, len_AC, len_BC,
                            alpha, conj_A, A, stride_A_AC,
                                   conj_B, B, stride_B_BC,
                             beta, conj_C, C, stride_C_AC, stride_C_BC);
            break;
        //TODO: gemv
        case HAS_AB+HAS_AC:
        case HAS_AB+HAS_BC:
        case HAS_AB+HAS_AC+HAS_BC:
            if (impl == BLAS_BASED)
            {
                contract_blas(comm, cfg, len_AB, len_AC, len_BC,
                              alpha, conj_A, A, stride_A_AB, stride_A_AC,
                                     conj_B, B, stride_B_AB, stride_B_BC,
                               beta, conj_C, C, stride_C_AC, stride_C_BC);
            }
            else
            {
                contract_blis(comm, cfg, len_AB, len_AC, len_BC,
                              alpha, conj_A, A, stride_A_AB, stride_A_AC,
                                     conj_B, B, stride_B_AB, stride_B_BC,
                               beta, conj_C, C, stride_C_AC, stride_C_BC);
            }
            break;
        case HAS_AB+HAS_ABC:
            while (iter_ABC.next(A, B, C))
            {
                dot(comm, cfg, len_AB, conj_A, A, stride_A_AB,
                                       conj_B, B, stride_B_AB, sum);
                if (beta == T(0))
                {
                    *C = alpha*sum;
                }
                else
                {
                    *C = alpha*sum + beta*(conj_C ? conj(*C) : *C);
                }
            }
            break;
        case HAS_AC+HAS_ABC:
            while (iter_ABC.next(A, B, C))
            {
                add(comm, cfg, {}, {}, len_AC, alpha*(conj_B ? conj(*B) : *B),
                    conj_A, A, {}, stride_A_AC, beta, conj_C, C, {}, stride_C_AC);
            }
            break;
        case HAS_BC+HAS_ABC:
            while (iter_ABC.next(A, B, C))
            {
                add(comm, cfg, {}, {}, len_BC, alpha*(conj_A ? conj(*A) : *A),
                    conj_B, B, {}, stride_B_BC, beta, conj_C, C, {}, stride_C_BC);
            }
            break;
        case HAS_AC+HAS_BC+HAS_ABC:
            weight_blas(comm, cfg, len_AC, len_BC, len_ABC,
                        alpha, conj_A, A, stride_A_AC, stride_A_ABC,
                               conj_B, B, stride_B_BC, stride_B_ABC,
                         beta, conj_C, C, stride_C_AC, stride_C_BC, stride_C_ABC);
            break;
        case HAS_AB+HAS_AC+HAS_ABC:
        case HAS_AB+HAS_BC+HAS_ABC:
        case HAS_AB+HAS_AC+HAS_BC+HAS_ABC:
            mult_blas(comm, cfg,
                      len_AB, len_AC, len_BC, len_ABC,
                      alpha, conj_A, A, stride_A_AB, stride_A_AC, stride_A_ABC,
                             conj_B, B, stride_B_AB, stride_B_BC, stride_B_ABC,
                       beta, conj_C, C, stride_C_AC, stride_C_BC, stride_C_ABC);
            break;
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, const config& cfg, \
                   const len_vector& len_AB, \
                   const len_vector& len_AC, \
                   const len_vector& len_BC, \
                   const len_vector& len_ABC, \
                   T alpha, bool conj_A, const T* A, \
                   const stride_vector& stride_A_AB, \
                   const stride_vector& stride_A_AC, \
                   const stride_vector& stride_A_ABC, \
                            bool conj_B, const T* B, \
                   const stride_vector& stride_B_AB, \
                   const stride_vector& stride_B_BC, \
                   const stride_vector& stride_B_ABC, \
                   T  beta, bool conj_C,       T* C, \
                   const stride_vector& stride_C_AC, \
                   const stride_vector& stride_C_BC, \
                   const stride_vector& stride_C_ABC);
#include "configs/foreach_type.h"

}
}
