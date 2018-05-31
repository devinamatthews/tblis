#include "cp_reform.hpp"
#include "khatri_rao.hpp"
#include "mult.hpp"

#include "util/tensor.hpp"

#include "memory/memory_pool.hpp"

#include "internal/3m/mult.hpp"

#include "internal/1t/dense/scale.hpp"
#include "internal/1t/dense/set.hpp"

namespace tblis
{

extern MemoryPool BuffersForA;

namespace internal
{

template <typename T>
void cp_reform(const communicator& comm, const config& cfg,
               const len_vector& len_m_, len_type len_r,
               T alpha, const ptr_vector<const T>& U_,
               const stride_vector& stride_U_m_,
               const stride_vector& stride_U_r_,
               T  beta, T* A_,
               const stride_vector& stride_A_m_)
{
    len_type m_tot = stl_ext::prod(len_m_);

    if (m_tot == 0) return;

    if (alpha == T(0) || len_r == 0)
    {
        if (beta == T(1))
        {
            set(comm, cfg, len_m_, T(0), A_, stride_A_m_);
        }
        else
        {
            scale(comm, cfg, len_m_, beta, false, A_, stride_A_m_);
        }

        return;
    }

    auto reorder_m = detail::sort_by_stride(stride_A_m_);

    auto U = stl_ext::permuted(U_, reorder_m);

    auto len_m = stl_ext::permuted(len_m_, reorder_m);
    unsigned ndim_m = len_m.size();

    auto stride_A_m = stl_ext::permuted(stride_A_m_, reorder_m);
    auto stride_U_m = stl_ext::permuted(stride_U_m_, reorder_m);
    auto stride_U_r = stl_ext::permuted(stride_U_r_, reorder_m);

    //
    // Get a buffer in which to store partial results.
    // Use the mc*kc "A" buffers that gemm uses because we will need one per
    // thread.
    //
    len_type workspace = cfg.gemm_kc.def<T>()*cfg.gemm_mc.def<T>();
    len_type r_step = std::min(workspace / (ndim_m - 2), len_r);

    for (len_type r_min = 0;r_min < len_r;r_min += r_step)
    {
        len_type r_max = std::min(len_r, r_min + r_step);

        comm.distribute_over_threads(m_tot/(len_m[0]*len_m[1]),
        [&](len_type m_min, len_type m_max)
        {
            auto buf = BuffersForA.allocate<T>(r_step * (ndim_m - 2));
            matrix_view<T> P({r_step, ndim_m - 2}, buf.template get<T>(), {1, r_step});

            viterator<1> iter_m(len_vector(len_m.begin()+2, len_m.end()),
                                stride_vector(stride_A_m.begin()+2, stride_A_m.end()));
            auto A = A_;
            iter_m.position(m_min, A);

            len_vector prev(ndim_m-2, -1);

            for (len_type m = m_min;m < m_max;m++)
            {
                iter_m.next(A);

                unsigned i = ndim_m;
                while (i --> 2)
                    if (iter_m.position()[i-2] != prev[i-2]) break;

                if (i == ndim_m-1)
                {
                    auto Ui = U[i] + iter_m.position()[i-2]*stride_U_m[i];

                    for (len_type r = r_min;r < r_max;r++)
                        P[r-r_min][i-2] = Ui[r*stride_U_r[i]];
                }
                else i++;

                while (i --> 2)
                {
                    auto Ui = U[i] + iter_m.position()[i-2]*stride_U_m[i];

                    for (len_type r = r_min;r < r_max;r++)
                        P[r-r_min][i-2] = Ui[r*stride_U_r[i]]*P[r-r_min][i-1];
                }

                prev = iter_m.position();

                auto U0 = U[0] + r_min*stride_U_r[0];
                auto U1 = U[1] + r_min*stride_U_r[1];
                auto P0 = &P[0][0];

                /*
                 * Compute A_ijkl... = U0_ir U1_jr U2_kr U3_lr ... using:
                 *
                 * for [kl...]:
                 *
                 *   P_r[kl...] = U2_kr U3_lr ... (partial KRP)
                 *
                 *   A_ij[kl...] = U0_ir P_r[kl...] U1_jr (diag-scaled MM)
                 */
                mult(single, cfg, len_m[0], len_m[1], r_max - r_min,
                     alpha, false, U0, stride_U_m[0], stride_U_r[0],
                            false, P0, 1,
                            false, U1, stride_U_r[1], stride_U_m[1],
                      beta, false,  A, stride_A_m[0], stride_A_m[1]);
            }
        });

        beta = T(1);
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void cp_reform(const communicator& comm, const config& cfg, \
                        const len_vector& len_m, len_type len_r, \
                        T alpha, const ptr_vector<const T>& U, \
                        const stride_vector& stride_U_m, \
                        const stride_vector& stride_U_r, \
                        T  beta, T* A, \
                        const stride_vector& stride_A_m);
#include "configs/foreach_type.h"

}
}
