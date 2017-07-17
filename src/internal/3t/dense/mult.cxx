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
                   T alpha, const T* A,
                   const stride_vector& stride_A_AB,
                   const stride_vector& stride_A_AC,
                            const T* B,
                   const stride_vector& stride_B_AB,
                   const stride_vector& stride_B_BC,
                   T  beta,       T* C,
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
            T(1), false,         A, {}, stride_A_AC+stride_A_AB,
            T(0), false, ar.data(), {},            ar.strides());

        add(comm, cfg, {}, {}, br.lengths(),
            T(1), false,         B, {}, stride_B_AB+stride_B_BC,
            T(0), false, br.data(), {},            br.strides());

        mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
             alpha, false, am.data(), am.stride(0), am.stride(1),
                    false, bm.data(), bm.stride(0), bm.stride(1),
              T(0), false, cm.data(), cm.stride(0), cm.stride(1));

        add(comm, cfg, {}, {}, cr.lengths(),
            T(1), false, cr.data(), {},            cr.strides(),
            beta, false,         C, {}, stride_C_AC+stride_C_BC);
    },
    ar, br, cr);
}

template <typename T>
void contract_ref(const communicator& comm, const config& cfg,
                  const len_vector& len_AB,
                  const len_vector& len_AC,
                  const len_vector& len_BC,
                  T alpha, const T* A,
                  const stride_vector& stride_A_AB,
                  const stride_vector& stride_A_AC,
                           const T* B,
                  const stride_vector& stride_B_AB,
                  const stride_vector& stride_B_BC,
                  T  beta,       T* C,
                  const stride_vector& stride_C_AC,
                  const stride_vector& stride_C_BC)
{
    (void)cfg;

    viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
    viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);
    len_type m = stl_ext::prod(len_AC);
    len_type n = stl_ext::prod(len_BC);

    len_type m_min, m_max, n_min, n_max;
    std::tie(m_min, m_max, std::ignore,
             n_min, n_max, std::ignore) = comm.distribute_over_threads_2d(m, n);

    const T* A0 = A;
    const T* B0 = B;
          T* C0 = C;

    iter_AC.position(m_min, A0, C0);

    for (len_type i = m_min;i < m_max;i++)
    {
        iter_AC.next(A0, C0);

        A = A0;
        B = B0;
        C = C0;

        iter_BC.position(n_min, B, C);

        for (len_type j = n_min;j < n_max;j++)
        {
            iter_BC.next(B, C);

            T temp = T();

            while (iter_AB.next(A, B))
            {
                temp += (*A)*(*B);
            }
            temp *= alpha;

            if (beta == T(0))
            {
                *C = temp;
            }
            else
            {
                *C = temp + beta*(*C);
            }
        }
    }
}

template <typename T>
void contract_blis(const communicator& comm, const config& cfg,
                   const len_vector& len_AB,
                   const len_vector& len_AC,
                   const len_vector& len_BC,
                   T alpha, const T* A,
                   const stride_vector& stride_A_AB,
                   const stride_vector& stride_A_AC,
                            const T* B,
                   const stride_vector& stride_B_AB,
                   const stride_vector& stride_B_BC,
                   T  beta,       T* C,
                   const stride_vector& stride_C_AC,
                   const stride_vector& stride_C_BC)
{
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
                            T alpha, const T* A, \
                            const stride_vector& stride_A_AB, \
                            const stride_vector& stride_A_AC, \
                                     const T* B, \
                            const stride_vector& stride_B_AB, \
                            const stride_vector& stride_B_BC, \
                            T  beta,       T* C, \
                            const stride_vector& stride_C_AC, \
                            const stride_vector& stride_C_BC);
#include "configs/foreach_type.h"

template <typename T>
void mult_blas(const communicator& comm, const config& cfg,
               const len_vector& len_AB,
               const len_vector& len_AC,
               const len_vector& len_BC,
               const len_vector& len_ABC,
               T alpha, const T* A,
               const stride_vector& stride_A_AB,
               const stride_vector& stride_A_AC,
               const stride_vector& stride_A_ABC,
                        const T* B,
               const stride_vector& stride_B_AB,
               const stride_vector& stride_B_BC,
               const stride_vector& stride_B_ABC,
               T  beta,       T* C,
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
                T(1), false,         A, {}, stride_A_AC+stride_A_AB,
                T(0), false, ar.data(), {},             ar.strides());

            add(comm, cfg, {}, {}, br.lengths(),
                T(1), false,         B, {}, stride_B_AB+stride_B_BC,
                T(0), false, br.data(), {},             br.strides());

            mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
                 alpha, false, am.data(), am.stride(0), am.stride(1),
                        false, bm.data(), bm.stride(0), bm.stride(1),
                  T(0), false, cm.data(), cm.stride(0), cm.stride(1));

            add(comm, cfg, {}, {}, cr.lengths(),
                T(1), false, cr.data(), {},            cr.strides(),
                beta, false,         C, {}, stride_C_AC+stride_C_BC);
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
              T alpha, const T* A,
              const stride_vector& stride_A_AB,
              const stride_vector& stride_A_AC,
              const stride_vector& stride_A_ABC,
                       const T* B,
              const stride_vector& stride_B_AB,
              const stride_vector& stride_B_BC,
              const stride_vector& stride_B_ABC,
              T  beta,       T* C,
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

                while (iter_AB.next(A, B))
                {
                    temp += (*A)*(*B);
                }

                temp *= alpha;

                if (beta == T(0))
                {
                    *C = temp;
                }
                else
                {
                    *C = temp + beta*(*C);
                }
            }
        }
    }
}

template <typename T>
void outer_prod_blas(const communicator& comm, const config& cfg,
                     const len_vector& len_AC,
                     const len_vector& len_BC,
                     T alpha, const T* A,
                     const stride_vector& stride_A_AC,
                              const T* B,
                     const stride_vector& stride_B_BC,
                     T  beta,       T* C,
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
            T(1), false,         A, {},  stride_A_AC,
            T(0), false, ar.data(), {}, ar.strides());

        add(comm, cfg, {}, {}, br.lengths(),
            T(1), false,         B, {},  stride_B_BC,
            T(0), false, br.data(), {}, br.strides());

        mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
             alpha, false, am.data(), am.stride(0), am.stride(1),
                    false, bm.data(), bm.stride(0), bm.stride(1),
              T(0), false, cm.data(), cm.stride(0), cm.stride(1));

        add(comm, cfg, {}, {}, cr.lengths(),
            T(1), false, cr.data(), {},             cr.strides(),
            beta, false,         C, {}, stride_C_AC+stride_C_BC);
    },
    ar, br, cr);
}

template <typename T>
void outer_prod_ref(const communicator& comm, const config& cfg,
                    const len_vector& len_AC,
                    const len_vector& len_BC,
                    T alpha, const T* A,
                    const stride_vector& stride_A_AC,
                             const T* B,
                    const stride_vector& stride_B_BC,
                    T  beta,       T* C,
                    const stride_vector& stride_C_AC,
                    const stride_vector& stride_C_BC)
{
    (void)cfg;

    viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);
    len_type m = stl_ext::prod(len_AC);
    len_type n = stl_ext::prod(len_BC);

    len_type m_min, m_max, n_min, n_max;
    std::tie(m_min, m_max, std::ignore,
             n_min, n_max, std::ignore) = comm.distribute_over_threads_2d(m, n);

    const T* A0 = A;
    const T* B0 = B;
          T* C0 = C;

    iter_AC.position(m_min, A0, C0);

    for (len_type i = m_min;i < m_max;i++)
    {
        iter_AC.next(A0, C0);

        A = A0;
        B = B0;
        C = C0;

        iter_BC.position(n_min, B, C);

        if (beta == T(0))
        {
            for (len_type j = n_min;j < n_max;j++)
            {
                iter_BC.next(B, C);
                *C = alpha*(*A)*(*B);
            }
        }
        else
        {
            for (len_type j = n_min;j < n_max;j++)
            {
                iter_BC.next(B, C);
                *C = alpha*(*A)*(*B) + beta*(*C);
            }
        }
    }
}

template <typename T>
void weight_blas(const communicator& comm, const config& cfg,
                 const len_vector& len_AC,
                 const len_vector& len_BC,
                 const len_vector& len_ABC,
                 T alpha, const T* A,
                 const stride_vector& stride_A_AC,
                 const stride_vector& stride_A_ABC,
                          const T* B,
                 const stride_vector& stride_B_BC,
                 const stride_vector& stride_B_ABC,
                 T  beta,       T* C,
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
                T(1), false,         A, {},  stride_A_AC,
                T(0), false, ar.data(), {}, ar.strides());

            add(comm, cfg, {}, {}, br.lengths(),
                T(1), false,         B, {},  stride_B_BC,
                T(0), false, br.data(), {}, br.strides());

            mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
                 alpha, false, am.data(), am.stride(0), am.stride(1),
                        false, bm.data(), bm.stride(0), bm.stride(1),
                  T(0), false, cm.data(), cm.stride(0), cm.stride(1));

            add(comm, cfg, {}, {}, cr.lengths(),
                T(1), false, cr.data(), {},             cr.strides(),
                beta, false,         C, {}, stride_C_AC+stride_C_BC);
        }
    },
    ar, br, cr);
}

template <typename T>
void weight_ref(const communicator& comm, const config& cfg,
                const len_vector& len_AC,
                const len_vector& len_BC,
                const len_vector& len_ABC,
                T alpha, const T* A,
                const stride_vector& stride_A_AC,
                const stride_vector& stride_A_ABC,
                         const T* B,
                const stride_vector& stride_B_BC,
                const stride_vector& stride_B_ABC,
                T  beta,       T* C,
                const stride_vector& stride_C_AC,
                const stride_vector& stride_C_BC,
                const stride_vector& stride_C_ABC)
{
    (void)cfg;

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
            if (beta == T(0))
            {
                while (iter_BC.next(B, C))
                {
                    *C = alpha*(*A)*(*B);
                }
            }
            else
            {
                while (iter_BC.next(B, C))
                {
                    *C = alpha*(*A)*(*B) + beta*(*C);
                }
            }
        }
    }
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
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C);

    if (len_AB.empty() && len_ABC.empty())
    {
        if (impl == REFERENCE)
        {
            outer_prod_ref(comm, cfg, len_AC, len_BC,
                           alpha, A, stride_A_AC,
                                  B, stride_B_BC,
                            beta, C, stride_C_AC, stride_C_BC);
        }
        else
        {
            outer_prod_blas(comm, cfg, len_AC, len_BC,
                            alpha, A, stride_A_AC,
                                   B, stride_B_BC,
                             beta, C, stride_C_AC, stride_C_BC);
        }
    }
    else if (len_AB.empty())
    {
        if (impl == REFERENCE || len_AC.empty() || len_BC.empty())
        {
            weight_ref(comm, cfg, len_AC, len_BC, len_ABC,
                       alpha, A, stride_A_AC, stride_A_ABC,
                              B, stride_B_BC, stride_B_ABC,
                        beta, C, stride_C_AC, stride_C_BC, stride_C_ABC);
        }
        else
        {
            weight_blas(comm, cfg, len_AC, len_BC, len_ABC,
                        alpha, A, stride_A_AC, stride_A_ABC,
                               B, stride_B_BC, stride_B_ABC,
                         beta, C, stride_C_AC, stride_C_BC, stride_C_ABC);
        }
    }
    else if (len_ABC.empty())
    {
        if (impl == REFERENCE)
        {
            contract_ref(comm, cfg, len_AB, len_AC, len_BC,
                         alpha, A, stride_A_AB, stride_A_AC,
                                B, stride_B_AB, stride_B_BC,
                          beta, C, stride_C_AC, stride_C_BC);
        }
        else if (impl == BLAS_BASED)
        {
            contract_blas(comm, cfg, len_AB, len_AC, len_BC,
                          alpha, A, stride_A_AB, stride_A_AC,
                                 B, stride_B_AB, stride_B_BC,
                           beta, C, stride_C_AC, stride_C_BC);
        }
        else
        {
            contract_blis(comm, cfg, len_AB, len_AC, len_BC,
                          alpha, A, stride_A_AB, stride_A_AC,
                                 B, stride_B_AB, stride_B_BC,
                           beta, C, stride_C_AC, stride_C_BC);
        }
    }
    else
    {
        if (impl == REFERENCE)
        {
            mult_ref(comm, cfg,
                     len_AB, len_AC, len_BC, len_ABC,
                     alpha, A, stride_A_AB, stride_A_AC, stride_A_ABC,
                            B, stride_B_AB, stride_B_BC, stride_B_ABC,
                      beta, C, stride_C_AC, stride_C_BC, stride_C_ABC);
        }
        else
        {
            mult_blas(comm, cfg,
                      len_AB, len_AC, len_BC, len_ABC,
                      alpha, A, stride_A_AB, stride_A_AC, stride_A_ABC,
                             B, stride_B_AB, stride_B_BC, stride_B_ABC,
                       beta, C, stride_C_AC, stride_C_BC, stride_C_ABC);
        }
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
