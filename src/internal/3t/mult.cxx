#include "mult.hpp"

#include "util/gemm_thread.hpp"
#include "util/tensor.hpp"

#include "nodes/matrify.hpp"
#include "nodes/partm.hpp"
#include "nodes/gemm_ukr.hpp"

#include "internal/1t/add.hpp"
#include "internal/3m/mult.hpp"

namespace tblis
{
namespace internal
{

impl_t impl = BLIS_BASED;

extern MemoryPool BuffersForA, BuffersForB, BuffersForScatter;
MemoryPool BuffersForScatter(4096);

using TensorGEMM = partition_gemm_nc<
                     partition_gemm_kc<
                       matrify_and_pack_b<BuffersForB,
                         partition_gemm_mc<
                           matrify_and_pack_a<BuffersForA,
                             matrify_c<BuffersForScatter,
                               partition_gemm_nr<
                                 partition_gemm_mr<
                                   gemm_micro_kernel>>>>>>>>;

template <typename T>
void contract_blas(const communicator& comm, const config& cfg,
                   const std::vector<len_type>& len_AB,
                   const std::vector<len_type>& len_AC,
                   const std::vector<len_type>& len_BC,
                   T alpha, const T* A,
                   const std::vector<stride_type>& stride_A_AB,
                   const std::vector<stride_type>& stride_A_AC,
                            const T* B,
                   const std::vector<stride_type>& stride_B_AB,
                   const std::vector<stride_type>& stride_B_BC,
                   T  beta,       T* C,
                   const std::vector<stride_type>& stride_C_AC,
                   const std::vector<stride_type>& stride_C_BC)
{
    tensor<T> ar, br, cr;
    T* ptrs_local[3];
    T** ptrs = &ptrs_local[0];

    if (comm.master())
    {
        ar.reset(len_AC+len_AB);
        br.reset(len_AB+len_BC);
        cr.reset(len_AC+len_BC);
        ptrs[0] = ar.data();
        ptrs[1] = br.data();
        ptrs[2] = cr.data();
    }

    comm.broadcast(ptrs);

    tensor_view<T> arv(len_AC+len_AB, ptrs[0]);
    tensor_view<T> brv(len_AB+len_BC, ptrs[1]);
    tensor_view<T> crv(len_AC+len_BC, ptrs[2]);

    matrix_view<T> am, bm, cm;
    matricize<T>(arv, am, static_cast<unsigned>(len_AC.size()));
    matricize<T>(brv, bm, static_cast<unsigned>(len_AB.size()));
    matricize<T>(crv, cm, static_cast<unsigned>(len_AC.size()));

    add(comm, cfg, {}, {}, arv.lengths(),
        T(1), false,          A, {}, stride_A_AC+stride_A_AB,
        T(0), false, arv.data(), {},           arv.strides());

    add(comm, cfg, {}, {}, brv.lengths(),
        T(1), false,          B, {}, stride_B_AB+stride_B_BC,
        T(0), false, brv.data(), {},           brv.strides());

    mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
         alpha, false, am.data(), am.stride(0), am.stride(1),
                false, bm.data(), bm.stride(0), bm.stride(1),
          T(0), false, cm.data(), cm.stride(0), cm.stride(1));

    add(comm, cfg, {}, {}, crv.lengths(),
        T(1), false, crv.data(), {},            crv.strides(),
        beta, false,          C, {}, stride_C_AC+stride_C_BC);
}

template <typename T>
void contract_ref(const communicator& comm, const config& cfg,
                  const std::vector<len_type>& len_AB,
                  const std::vector<len_type>& len_AC,
                  const std::vector<len_type>& len_BC,
                  T alpha, const T* A,
                  const std::vector<stride_type>& stride_A_AB,
                  const std::vector<stride_type>& stride_A_AC,
                           const T* B,
                  const std::vector<stride_type>& stride_B_AB,
                  const std::vector<stride_type>& stride_B_BC,
                  T  beta,       T* C,
                  const std::vector<stride_type>& stride_C_AC,
                  const std::vector<stride_type>& stride_C_BC)
{
    (void)cfg;

    MArray::viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
    MArray::viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    MArray::viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);
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
                   const std::vector<len_type>& len_AB,
                   const std::vector<len_type>& len_AC,
                   const std::vector<len_type>& len_BC,
                   T alpha, const T* A,
                   const std::vector<stride_type>& stride_A_AB,
                   const std::vector<stride_type>& stride_A_AC,
                            const T* B,
                   const std::vector<stride_type>& stride_B_AB,
                   const std::vector<stride_type>& stride_B_BC,
                   T  beta,       T* C,
                   const std::vector<stride_type>& stride_C_AC,
                   const std::vector<stride_type>& stride_C_BC)
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

    const bool row_major = cfg.gemm_row_major<T>();

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

    int nt = comm.num_threads();
    auto tc = make_gemm_thread_config(nt, m, n, k);
    step<0>(gemm).distribute = tc.jc_nt;
    step<4>(gemm).distribute = tc.ic_nt;
    step<8>(gemm).distribute = tc.jr_nt;
    step<9>(gemm).distribute = tc.ir_nt;

    gemm(comm, cfg, alpha, at, bt, beta, ct);
}

template <typename T>
void mult_blas(const communicator& comm, const config& cfg,
               const std::vector<len_type>& len_A,
               const std::vector<len_type>& len_B,
               const std::vector<len_type>& len_C,
               const std::vector<len_type>& len_AB,
               const std::vector<len_type>& len_AC,
               const std::vector<len_type>& len_BC,
               const std::vector<len_type>& len_ABC,
               T alpha, const T* A,
               const std::vector<stride_type>& stride_A_A,
               const std::vector<stride_type>& stride_A_AB,
               const std::vector<stride_type>& stride_A_AC,
               const std::vector<stride_type>& stride_A_ABC,
                        const T* B,
               const std::vector<stride_type>& stride_B_B,
               const std::vector<stride_type>& stride_B_AB,
               const std::vector<stride_type>& stride_B_BC,
               const std::vector<stride_type>& stride_B_ABC,
               T  beta,       T* C,
               const std::vector<stride_type>& stride_C_C,
               const std::vector<stride_type>& stride_C_AC,
               const std::vector<stride_type>& stride_C_BC,
               const std::vector<stride_type>& stride_C_ABC)
{
    tensor<T> ar, br, cr;
    T* ptrs_local[3];
    T** ptrs = &ptrs_local[0];

    if (comm.master())
    {
        ar.reset(len_AC+len_AB);
        br.reset(len_AB+len_BC);
        cr.reset(len_AC+len_BC);
        ptrs[0] = ar.data();
        ptrs[1] = br.data();
        ptrs[2] = cr.data();
    }

    comm.broadcast(ptrs);

    tensor_view<T> arv(len_AC+len_AB, ptrs[0]);
    tensor_view<T> brv(len_AB+len_BC, ptrs[1]);
    tensor_view<T> crv(len_AC+len_BC, ptrs[2]);

    matrix_view<T> am, bm, cm;
    matricize<T>(arv, am, static_cast<unsigned>(len_AC.size()));
    matricize<T>(brv, bm, static_cast<unsigned>(len_AB.size()));
    matricize<T>(crv, cm, static_cast<unsigned>(len_AC.size()));

    MArray::viterator<3> it(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

    while (it.next(A, B, C))
    {
        add(comm, cfg, len_A, {}, arv.lengths(),
            T(1), false,          A, stride_A_A, stride_A_AC+stride_A_AB,
            T(0), false, arv.data(),         {},           arv.strides());

        add(comm, cfg, len_B, {}, brv.lengths(),
            T(1), false,          B, stride_B_B, stride_B_AB+stride_B_BC,
            T(0), false, brv.data(),         {},           brv.strides());

        mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
             alpha, false, am.data(), am.stride(0), am.stride(1),
                    false, bm.data(), bm.stride(0), bm.stride(1),
              T(0), false, cm.data(), cm.stride(0), cm.stride(1));

        add(comm, cfg, {}, len_C, crv.lengths(),
            T(1), false, crv.data(),         {},            crv.strides(),
            beta, false,          C, stride_C_C, stride_C_AC+stride_C_BC);
    }
}

template <typename T>
void mult_ref(const communicator& comm, const config& cfg,
              const std::vector<len_type>& len_A,
              const std::vector<len_type>& len_B,
              const std::vector<len_type>& len_C,
              const std::vector<len_type>& len_AB,
              const std::vector<len_type>& len_AC,
              const std::vector<len_type>& len_BC,
              const std::vector<len_type>& len_ABC,
              T alpha, const T* A,
              const std::vector<stride_type>& stride_A_A,
              const std::vector<stride_type>& stride_A_AB,
              const std::vector<stride_type>& stride_A_AC,
              const std::vector<stride_type>& stride_A_ABC,
                       const T* B,
              const std::vector<stride_type>& stride_B_B,
              const std::vector<stride_type>& stride_B_AB,
              const std::vector<stride_type>& stride_B_BC,
              const std::vector<stride_type>& stride_B_ABC,
              T  beta,       T* C,
              const std::vector<stride_type>& stride_C_C,
              const std::vector<stride_type>& stride_C_AC,
              const std::vector<stride_type>& stride_C_BC,
              const std::vector<stride_type>& stride_C_ABC)
{
    (void)cfg;

    MArray::viterator<1> iter_A(len_A, stride_A_A);
    MArray::viterator<1> iter_B(len_B, stride_B_B);
    MArray::viterator<1> iter_C(len_C, stride_C_C);
    MArray::viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
    MArray::viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    MArray::viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);
    MArray::viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);
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
                    T temp_A = T();
                    while (iter_A.next(A))
                    {
                        temp_A += *A;
                    }

                    T temp_B = T();
                    while (iter_B.next(B))
                    {
                        temp_B += *B;
                    }

                    temp += temp_A*temp_B;
                }

                temp *= alpha;

                if (beta == T(0))
                {
                    while (iter_C.next(C))
                    {
                        *C = temp;
                    }
                }
                else
                {
                    while (iter_C.next(C))
                    {
                        *C = temp + beta*(*C);
                    }
                }
            }
        }
    }
}

template <typename T>
void outer_prod_blas(const communicator& comm, const config& cfg,
                     const std::vector<len_type>& len_AC,
                     const std::vector<len_type>& len_BC,
                     T alpha, const T* A,
                     const std::vector<stride_type>& stride_A_AC,
                              const T* B,
                     const std::vector<stride_type>& stride_B_BC,
                     T  beta,       T* C,
                     const std::vector<stride_type>& stride_C_AC,
                     const std::vector<stride_type>& stride_C_BC)
{
    tensor<T> ar, br, cr;
    T* ptrs_local[3];
    T** ptrs = &ptrs_local[0];

    if (comm.master())
    {
        ar.reset(len_AC);
        br.reset(len_BC);
        cr.reset(len_AC+len_BC);
        ptrs[0] = ar.data();
        ptrs[1] = br.data();
        ptrs[2] = cr.data();
    }

    comm.broadcast(ptrs);

    tensor_view<T> arv(len_AC, ptrs[0]);
    tensor_view<T> brv(len_BC, ptrs[1]);
    tensor_view<T> crv(len_AC+len_BC, ptrs[2]);

    matrix_view<T> am, bm, cm;
    matricize<T>(arv, am, static_cast<unsigned>(len_AC.size()));
    matricize<T>(brv, bm, 0);
    matricize<T>(crv, cm, static_cast<unsigned>(len_AC.size()));

    add(comm, cfg, {}, {}, arv.lengths(),
        T(1), false,          A, {},   stride_A_AC,
        T(0), false, arv.data(), {}, arv.strides());

    add(comm, cfg, {}, {}, brv.lengths(),
        T(1), false,          B, {},   stride_B_BC,
        T(0), false, brv.data(), {}, brv.strides());

    mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
         alpha, false, am.data(), am.stride(0), am.stride(1),
                false, bm.data(), bm.stride(0), bm.stride(1),
          T(0), false, cm.data(), cm.stride(0), cm.stride(1));

    add(comm, cfg, {}, {}, crv.lengths(),
        T(1), false, crv.data(), {},            crv.strides(),
        beta, false,          C, {}, stride_C_AC+stride_C_BC);
}

template <typename T>
void outer_prod_ref(const communicator& comm, const config& cfg,
                    const std::vector<len_type>& len_AC,
                    const std::vector<len_type>& len_BC,
                    T alpha, const T* A,
                    const std::vector<stride_type>& stride_A_AC,
                             const T* B,
                    const std::vector<stride_type>& stride_B_BC,
                    T  beta,       T* C,
                    const std::vector<stride_type>& stride_C_AC,
                    const std::vector<stride_type>& stride_C_BC)
{
    (void)cfg;

    MArray::viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    MArray::viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);
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
                 const std::vector<len_type>& len_AC,
                 const std::vector<len_type>& len_BC,
                 const std::vector<len_type>& len_ABC,
                 T alpha, const T* A,
                 const std::vector<stride_type>& stride_A_AC,
                 const std::vector<stride_type>& stride_A_ABC,
                          const T* B,
                 const std::vector<stride_type>& stride_B_BC,
                 const std::vector<stride_type>& stride_B_ABC,
                 T  beta,       T* C,
                 const std::vector<stride_type>& stride_C_AC,
                 const std::vector<stride_type>& stride_C_BC,
                 const std::vector<stride_type>& stride_C_ABC)
{
    tensor<T> ar, br, cr;
    T* ptrs_local[3];
    T** ptrs = &ptrs_local[0];

    if (comm.master())
    {
        ar.reset(len_AC);
        br.reset(len_BC);
        cr.reset(len_AC+len_BC);
        ptrs[0] = ar.data();
        ptrs[1] = br.data();
        ptrs[2] = cr.data();
    }

    comm.broadcast(ptrs);

    tensor_view<T> arv(len_AC, ptrs[0]);
    tensor_view<T> brv(len_BC, ptrs[1]);
    tensor_view<T> crv(len_AC+len_BC, ptrs[2]);

    matrix_view<T> am, bm, cm;
    matricize<T>(arv, am, static_cast<unsigned>(len_AC.size()));
    matricize<T>(brv, bm, 0);
    matricize<T>(crv, cm, static_cast<unsigned>(len_AC.size()));

    MArray::viterator<3> it(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

    while (it.next(A, B, C))
    {
        add(comm, cfg, {}, {}, arv.lengths(),
            T(1), false,          A, {},   stride_A_AC,
            T(0), false, arv.data(), {}, arv.strides());

        add(comm, cfg, {}, {}, brv.lengths(),
            T(1), false,          B, {},   stride_B_BC,
            T(0), false, brv.data(), {}, brv.strides());

        mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
             alpha, false, am.data(), am.stride(0), am.stride(1),
                    false, bm.data(), bm.stride(0), bm.stride(1),
              T(0), false, cm.data(), cm.stride(0), cm.stride(1));

        add(comm, cfg, {}, {}, crv.lengths(),
            T(1), false, crv.data(), {},            crv.strides(),
            beta, false,          C, {}, stride_C_AC+stride_C_BC);
    }
}

template <typename T>
void weight_ref(const communicator& comm, const config& cfg,
                const std::vector<len_type>& len_AC,
                const std::vector<len_type>& len_BC,
                const std::vector<len_type>& len_ABC,
                T alpha, const T* A,
                const std::vector<stride_type>& stride_A_AC,
                const std::vector<stride_type>& stride_A_ABC,
                         const T* B,
                const std::vector<stride_type>& stride_B_BC,
                const std::vector<stride_type>& stride_B_ABC,
                T  beta,       T* C,
                const std::vector<stride_type>& stride_C_AC,
                const std::vector<stride_type>& stride_C_BC,
                const std::vector<stride_type>& stride_C_ABC)
{
    (void)cfg;

    MArray::viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    MArray::viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);
    MArray::viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);
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
          const std::vector<len_type>& len_A,
          const std::vector<len_type>& len_B,
          const std::vector<len_type>& len_C,
          const std::vector<len_type>& len_AB,
          const std::vector<len_type>& len_AC,
          const std::vector<len_type>& len_BC,
          const std::vector<len_type>& len_ABC,
          T alpha, bool conj_A, const T* A,
          const std::vector<stride_type>& stride_A_A,
          const std::vector<stride_type>& stride_A_AB,
          const std::vector<stride_type>& stride_A_AC,
          const std::vector<stride_type>& stride_A_ABC,
                   bool conj_B, const T* B,
          const std::vector<stride_type>& stride_B_B,
          const std::vector<stride_type>& stride_B_AB,
          const std::vector<stride_type>& stride_B_BC,
          const std::vector<stride_type>& stride_B_ABC,
          T  beta, bool conj_C,       T* C,
          const std::vector<stride_type>& stride_C_C,
          const std::vector<stride_type>& stride_C_AC,
          const std::vector<stride_type>& stride_C_BC,
          const std::vector<stride_type>& stride_C_ABC)
{
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C);

    if (len_A.empty() && len_B.empty() && len_C.empty() &&
        (len_AB.empty() || len_ABC.empty()))
    {
        if (len_AB.empty())
        {
            if (len_ABC.empty())
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
            else
            {
                if (impl == REFERENCE)
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
        }
        else
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
    }
    else
    {
        if (impl == REFERENCE)
        {
            mult_ref(comm, cfg, len_A, len_B, len_C,
                     len_AB, len_AC, len_BC, len_ABC,
                     alpha, A, stride_A_A, stride_A_AB,
                               stride_A_AC, stride_A_ABC,
                            B, stride_B_B, stride_B_AB,
                               stride_B_BC, stride_B_ABC,
                      beta, C, stride_C_C, stride_C_AC,
                               stride_C_BC, stride_C_ABC);
        }
        else
        {
            mult_blas(comm, cfg, len_A, len_B, len_C,
                      len_AB, len_AC, len_BC, len_ABC,
                      alpha, A, stride_A_A, stride_A_AB,
                                stride_A_AC, stride_A_ABC,
                             B, stride_B_B, stride_B_AB,
                                stride_B_BC, stride_B_ABC,
                       beta, C, stride_C_C, stride_C_AC,
                                stride_C_BC, stride_C_ABC);
        }
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, const config& cfg, \
                   const std::vector<len_type>& len_A, \
                   const std::vector<len_type>& len_B, \
                   const std::vector<len_type>& len_C, \
                   const std::vector<len_type>& len_AB, \
                   const std::vector<len_type>& len_AC, \
                   const std::vector<len_type>& len_BC, \
                   const std::vector<len_type>& len_ABC, \
                   T alpha, bool conj_A, const T* A, \
                   const std::vector<stride_type>& stride_A_A, \
                   const std::vector<stride_type>& stride_A_AB, \
                   const std::vector<stride_type>& stride_A_AC, \
                   const std::vector<stride_type>& stride_A_ABC, \
                            bool conj_B, const T* B, \
                   const std::vector<stride_type>& stride_B_B, \
                   const std::vector<stride_type>& stride_B_AB, \
                   const std::vector<stride_type>& stride_B_BC, \
                   const std::vector<stride_type>& stride_B_ABC, \
                   T  beta, bool conj_C,       T* C, \
                   const std::vector<stride_type>& stride_C_C, \
                   const std::vector<stride_type>& stride_C_AC, \
                   const std::vector<stride_type>& stride_C_BC, \
                   const std::vector<stride_type>& stride_C_ABC);
#include "configs/foreach_type.h"

}
}
