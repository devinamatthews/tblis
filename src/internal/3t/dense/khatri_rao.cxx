#include "khatri_rao.hpp"
#include "mult.hpp"

#include "util/tensor.hpp"

#include "memory/memory_pool.hpp"

#include "internal/1t/dense/add.hpp"
#include "internal/1t/dense/scale.hpp"
#include "internal/1t/dense/set.hpp"

namespace tblis
{

extern MemoryPool BuffersForA;

namespace internal
{

template <typename T>
void khatri_rao(const communicator& comm, const config& cfg,
                const len_vector& len_m_, len_type len_r,
                T alpha, const ptr_vector<const T>& U_,
                const stride_vector& stride_U_m_,
                const stride_vector& stride_U_r_,
                T  beta, T* A,
                const stride_vector& stride_A_m_,
                stride_type stride_A_r)
{
    len_type m_tot = stl_ext::prod(len_m_);

    if (m_tot == 0 || len_r == 0) return;

    if (alpha == T(0))
    {
        auto len_A = len_m_;
        len_A.push_back(len_r);

        auto stride_A = stride_A_m_;
        stride_A.push_back(stride_A_r);

        if (beta == T(1))
        {
            set(comm, cfg, len_A, T(0), A, stride_A);
        }
        else
        {
            scale(comm, cfg, len_A, beta, false, A, stride_A);
        }

        return;
    }

    unsigned ndim_m = len_m_.size();

    if (ndim_m == 1)
    {
        add(comm, cfg, {}, {}, {len_m_[0], len_r},
            alpha, false, U_[0], {}, {stride_U_m_[0], stride_U_r_[0]},
             beta, false,     A, {}, {stride_A_m_[0], stride_A_r});
        return;
    }
    else if (ndim_m == 2)
    {
        mult(comm, cfg, {}, {len_m_[0]}, {len_m_[1]}, {len_r},
             alpha, false, U_[0], {}, {stride_U_m_[0]}, {stride_U_r_[0]},
                    false, U_[1], {}, {stride_U_m_[1]}, {stride_U_r_[1]},
              beta, false,     A, {stride_A_m_[0]}, {stride_A_m_[1]}, {stride_A_r});
        return;
    }

    auto reorder_m = detail::sort_by_stride(stride_A_m_);

    auto U = stl_ext::permuted(U_, reorder_m);

    auto len_m = stl_ext::permuted(len_m_, reorder_m);

    auto stride_A_m = stl_ext::permuted(stride_A_m_, reorder_m);
    auto stride_U_m = stl_ext::permuted(stride_U_m_, reorder_m);
    auto stride_U_r = stl_ext::permuted(stride_U_r_, reorder_m);

    //
    // Get a buffer in which to store partial results.
    // Use the mc*kc "A" buffers that gemm uses because we will need one per
    // thread.
    //
    len_type workspace = cfg.gemm_kc.def<T>()*cfg.gemm_mc.def<T>();
    len_type r_step = std::min(workspace / (ndim_m - 1), len_r);

    for (len_type r_min = 0;r_min < len_r;r_min += r_step)
    {
        len_type r_max = std::min(len_r, r_min + r_step);

        comm.distribute_over_threads(m_tot/len_m[0],
        [&](len_type m_min, len_type m_max)
        {
            auto buf = BuffersForA.allocate<T>(r_step * (ndim_m - 1));
            matrix_view<T> P({r_step, ndim_m - 1}, buf.template get<T>(), {1, r_step});

            viterator<1> iter_m(len_vector(len_m.begin()+1, len_m.end()),
                                stride_vector(stride_A_m.begin()+1, stride_A_m.end()));
            iter_m.position(m_min, A);

            len_vector prev(ndim_m-1, -1);

            for (len_type m = m_min;m < m_max;m++)
            {
                iter_m.next(A);

                unsigned i = ndim_m;
                while (i --> 1)
                    if (iter_m.position()[i-1] != prev[i-1]) break;

                if (i == ndim_m-1)
                {
                    auto Ui = U[i] + iter_m.position()[i-1]*stride_U_m[i];

                    for (len_type r = r_min;r < r_max;r++)
                        P[r-r_min][i-1] = Ui[r*stride_U_r[i]];
                }
                else i++;

                while (i --> 1)
                {
                    auto Ui = U[i] + iter_m.position()[i-1]*stride_U_m[i];

                    for (len_type r = r_min;r < r_max;r++)
                        P[r-r_min][i-1] = Ui[r*stride_U_r[i]]*P[r-r_min][i];
                }

                prev = iter_m.position();

                auto A0 = A + r_min*stride_A_r;
                auto U0 = U[0] + r_min*stride_U_r[0];
                auto P0 = &P[0][0];

                if (stride_A_r == 1)
                {
                    for (len_type m0 = 0;m0 < len_m[0];m0++)
                    {
                        cfg.mult_ukr.call<T>(r_max - r_min,
                                             alpha, false, U0, stride_U_r[0],
                                                    false, P0,             1,
                                              beta, false, A0,             1);

                        A0 += stride_A_m[0];
                        U0 += stride_U_m[0];
                    }
                }
                else
                {
                    for (len_type r = 0;r < r_max - r_min;r++)
                    {
                        cfg.add_ukr.call<T>(len_m[0],
                                            alpha*P0[r], false, U0, stride_U_m[0],
                                                   beta, false, A0, stride_A_m[0]);

                        A0 += stride_A_r;
                        U0 += stride_U_r[0];
                    }
                }
            }
        });
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void khatri_rao(const communicator& comm, const config& cfg, \
                         const len_vector& len_m, len_type len_r, \
                         T alpha, const ptr_vector<const T>& U, \
                         const stride_vector& stride_U_m, \
                         const stride_vector& stride_U_r, \
                         T  beta, T* A, \
                         const stride_vector& stride_A_m, \
                         stride_type stride_A_r);
#include "configs/foreach_type.h"

}
}
