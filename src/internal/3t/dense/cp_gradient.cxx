#include "cp_gradient.hpp"
#include "khatri_rao.hpp"
#include "mult.hpp"

#include "util/tensor.hpp"

#include "memory/memory_pool.hpp"

#include "internal/1m/add.hpp"
#include "internal/1m/set.hpp"

#include "internal/3m/mult.hpp"

namespace tblis
{
namespace internal
{

cp_impl_t cp_impl = DIRECT;

template <typename T>
void cp_gradient_naive(const communicator& comm, const config& cfg,
                       const len_vector& len_m, len_type len_n, len_type len_r,
                       const T* A,
                       const stride_vector& stride_A_m, stride_type stride_A_n,
                       const ptr_vector<const T>& U,
                       const stride_vector& stride_U_m,
                       const stride_vector& stride_U_r,
                       T* G, stride_type stride_G_n, stride_type stride_G_r)
{
    auto len_K = len_m;
    len_K.push_back(len_r);
    varray<T> K(len_K, uninitialized, COLUMN_MAJOR);

    stride_vector stride_K_m(K.strides().begin(), K.strides().end()-1);
    stride_type stride_K_r = K.strides().back();

    /*
     * K_ijk...r = U_ir U_jr U_kr ...
     */
    khatri_rao(comm, cfg, len_m, len_r,
               T(1), U, stride_U_m, stride_U_r,
               T(0), K.data(), stride_K_m, stride_K_r);

    /*
     * G_nr = A_ijk...n K_ijk...r
     *
     *      = A_ijk...n U_ir U_jr U_kr ...
     */
    mult(comm, cfg, len_m, {len_n}, {len_r}, {},
         T(1), false,        A, stride_A_m, {stride_A_n}, {},
               false, K.data(), stride_K_m, {stride_K_r}, {},
         T(0), false,        G, {stride_G_n}, {stride_G_r}, {});
}

template <typename T>
void cp_gradient_phan(const communicator& comm, const config& cfg,
                      const len_vector& len_m, len_type len_n, len_type len_r,
                      const T* A,
                      const stride_vector& stride_A_m, stride_type stride_A_n,
                      const ptr_vector<const T>& U,
                      const stride_vector& stride_U_m,
                      const stride_vector& stride_U_r,
                      T* G, stride_type stride_G_n, stride_type stride_G_r)
{
    len_type m_tot = stl_ext::prod(len_m);
    unsigned ndim_m = len_m.size();

    /*
     * Find the point k at which len_r = len_m[:k] ~ len_l = len_m[k:]
     */
    len_type num_L = m_tot;
    len_type num_R = 1;
    unsigned ndim_R = 0;
    for (;ndim_R < ndim_m;ndim_R++)
    {
        len_type num_R2 = num_R*len_m[ndim_R];
        len_type num_L2 = num_L/len_m[ndim_R];

        /*
         * We start with num_L > num_R, so if in the next iteration
         * num_R > num_L, then we have reached the balance point
         */
        if (num_R2 > num_L2)
        {
            /*
             * But, we have to check whether the current point or the next
             * one is "closer" to equality. Do this by looking at the ratio
             * num_L:num_R.
             *
             * NB: R2/L2 < L/R => R2*R2 < L*L
             */
            if (num_R2*num_R2 < num_L*num_L)
            {
                ndim_R++;
                num_R = num_R2;
                num_L = num_L2;
            }

            break;
        }

        num_R = num_R2;
        num_L = num_L2;
    }

    if (ndim_R == 0 || ndim_R == ndim_m)
    {
        cp_gradient_naive(comm, cfg, len_m, len_n, len_r,
                          A, stride_A_m, stride_A_n,
                          U, stride_U_m, stride_U_r,
                          G, stride_G_n, stride_G_r);
        return;
    }

    len_vector len_R_m(len_m.begin(), len_m.begin()+ndim_R);
    len_vector len_L_m(len_m.begin()+ndim_R, len_m.end());

    stride_vector stride_AL_m(stride_A_m.begin()+ndim_R, stride_A_m.end());
    stride_vector stride_AR_m(stride_A_m.begin(), stride_A_m.begin()+ndim_R);
    stride_vector stride_UL_m(stride_U_m.begin()+ndim_R, stride_U_m.end());
    stride_vector stride_UR_m(stride_U_m.begin(), stride_U_m.begin()+ndim_R);
    stride_vector stride_UL_r(stride_U_r.begin()+ndim_R, stride_U_r.end());
    stride_vector stride_UR_r(stride_U_r.begin(), stride_U_r.begin()+ndim_R);
    ptr_vector<const T> UL(U.begin()+ndim_R, U.end());
    ptr_vector<const T> UR(U.begin(), U.begin()+ndim_R);

    len_R_m.push_back(len_r);
    varray<T> R(len_R_m, uninitialized, COLUMN_MAJOR);
    len_R_m.pop_back();

    stride_vector stride_R_m(R.strides().begin(), R.strides().end()-1);
    stride_type stride_R_r(R.strides().back());

    /*
     * R_kl...r = U_kr U_lr ...
     */
    khatri_rao(comm, cfg, len_R_m, len_r,
               T(1), UR, stride_UR_m, stride_UR_r,
               T(0), R.data(), stride_R_m, stride_R_r);

    len_vector len_P = len_L_m;
    len_P.push_back(len_n);
    len_P.push_back(len_r);
    varray<T> P(len_P, uninitialized, COLUMN_MAJOR);

    stride_vector stride_P_m(P.strides().begin(), P.strides().end()-2);
    stride_type stride_P_n = P.strides()[P.dimension()-2];
    stride_type stride_P_r = P.strides()[P.dimension()-1];

    len_L_m.push_back(len_n);
    stride_AL_m.push_back(stride_A_n);
    stride_P_m.push_back(stride_P_n);

    /*
     * P_ij...nr = A_ij...kl...n R_kl...r
     */
    mult(comm, cfg, len_R_m, len_L_m, {len_r}, {},
         T(1), false,        A, stride_AR_m, stride_AL_m, {},
               false, R.data(), stride_R_m, {stride_R_r}, {},
         T(0), false, P.data(), stride_P_m, {stride_P_r}, {});

    len_L_m.pop_back();
    stride_AL_m.pop_back();
    stride_P_m.pop_back();

    R.reset();

    len_L_m.push_back(len_r);
    varray<T> L(len_L_m, uninitialized, COLUMN_MAJOR);
    len_L_m.pop_back();

    stride_vector stride_L_m(L.strides().begin(), L.strides().end()-1);
    stride_type stride_L_r(L.strides().back());

    /*
     * L_ij...r = U_ir U_jr ...
     */
    khatri_rao(comm, cfg, len_L_m, len_r,
               T(1), UL, stride_UL_m, stride_UL_r,
               T(0), L.data(), stride_L_m, stride_L_r);

    /*
     * G_nr = P_ij...nr L_ij...r
     *
     *      = A_ij...kl...n U_ir U_jr ... U_kr U_lr ...
     */
    mult(comm, cfg, len_L_m, {len_n}, {}, {len_r},
         T(1), false, P.data(), stride_P_m, {stride_P_n}, {stride_P_r},
               false, L.data(), stride_L_m, {}, {stride_L_r},
         T(0), false,        G, {stride_G_n}, {}, {stride_G_r});
}

template <typename T>
void cp_gradient_direct(const communicator& comm, const config& cfg,
                        const len_vector& len_m, len_type len_n, len_type len_r,
                        const T* A,
                        const stride_vector& stride_A_m, stride_type stride_A_n,
                        const ptr_vector<const T>& U,
                        const stride_vector& stride_U_m,
                        const stride_vector& stride_U_r,
                        T* G, stride_type stride_G_n, stride_type stride_G_r)
{
    len_type m_tot = stl_ext::prod(len_m);
    unsigned ndim_m = len_m.size();

    /*
     * G_nr = A_ijk...n U_ir U_jr U_kr ...
     *
     *      = A_in[jk...] U_ir (U_r[j] U_r[k] ...)
     */

    set(comm, cfg, len_n, len_r,
        T(0), G, stride_G_n, stride_G_r);

    tci::mutex G_lock;
    comm.broadcast(
    [&](tci::mutex& G_lock)
    {
        comm.distribute_over_threads(m_tot/len_m[0],
        [&](len_type m_min, len_type m_max)
        {
            matrix<T> P({len_r, ndim_m - 1}, uninitialized, COLUMN_MAJOR);
            matrix<T> G_local;

            if (stride_G_n < stride_G_r)
                G_local.reset({len_n, len_r}, uninitialized, COLUMN_MAJOR);
            else
                G_local.reset({len_n, len_r}, uninitialized, ROW_MAJOR);

            viterator<1> iter_m(len_vector(len_m.begin()+1, len_m.end()),
                                stride_vector(stride_A_m.begin()+1, stride_A_m.end()));
            iter_m.position(m_min, A);

            len_vector prev(ndim_m-1, -1);

            T beta = T(0);
            for (len_type m = m_min;m < m_max;m++)
            {
                iter_m.next(A);

                unsigned i = ndim_m;
                while (i --> 1)
                    if (iter_m.position()[i-1] != prev[i-1]) break;

                if (i == ndim_m-1)
                {
                    auto Ui = U[i] + iter_m.position()[i-1]*stride_U_m[i];

                    for (len_type r = 0;r < len_r;r++)
                        P[r][i-1] = Ui[r*stride_U_r[i]];
                }
                else i++;

                while (i --> 1)
                {
                    auto Ui = U[i] + iter_m.position()[i-1]*stride_U_m[i];

                    for (len_type r = 0;r < len_r;r++)
                        P[r][i-1] = Ui[r*stride_U_r[i]]*P[r][i];
                }

                prev = iter_m.position();

                mult(single, cfg, len_n, len_r, len_m[0],
                     T(1), false, A, stride_A_n, stride_A_m[0],
                           false, U[0], stride_U_m[0], stride_U_r[0],
                           false, &P[0][0], 1,
                     beta, false, G_local.data(), G_local.stride(0), G_local.stride(1));

                beta = T(1);
            }

            std::lock_guard<tci::mutex> guard(G_lock);

            add(single, cfg, len_n, len_r,
                T(1), false, G_local.data(), G_local.stride(0), G_local.stride(1),
                T(1), false, G, stride_G_n, stride_G_r);
        });
    }, G_lock);

    comm.barrier();
}

template <typename T>
void cp_gradient(const communicator& comm, const config& cfg,
                 const len_vector& len_m_, len_type len_n, len_type len_r,
                 const T* A,
                 const stride_vector& stride_A_m_, stride_type stride_A_n,
                 const ptr_vector<const T>& U_,
                 const stride_vector& stride_U_m_,
                 const stride_vector& stride_U_r_,
                 T* G, stride_type stride_G_n, stride_type stride_G_r)
{
    len_type m_tot = stl_ext::prod(len_m_);

    if (m_tot == 0 || len_n == 0) return;

    if (len_r == 0)
    {
        set(comm, cfg, len_r, len_n, T(0), G, stride_G_n, stride_G_r);
        return;
    }

    if (len_m_.size() == 1)
    {
        mult(comm, cfg, len_n, len_r, len_m_[0],
             T(1), false, A, stride_A_n, stride_A_m_[0],
                   false, U_[0], stride_U_m_[0], stride_U_r_[0],
             T(0), false, G, stride_G_n, stride_G_r);
        return;
    }

    auto reorder_m = detail::sort_by_stride(stride_A_m_);

    auto U = stl_ext::permuted(U_, reorder_m);

    auto len_m = stl_ext::permuted(len_m_, reorder_m);

    auto stride_A_m = stl_ext::permuted(stride_A_m_, reorder_m);
    auto stride_U_m = stl_ext::permuted(stride_U_m_, reorder_m);
    auto stride_U_r = stl_ext::permuted(stride_U_r_, reorder_m);

    if (cp_impl == DIRECT)
    {
        cp_gradient_direct(comm, cfg, len_m, len_n, len_r,
                           A, stride_A_m, stride_A_n,
                           U, stride_U_m, stride_U_r,
                           G, stride_G_n, stride_G_r);
    }
    else if (cp_impl == PHAN)
    {
        cp_gradient_phan(comm, cfg, len_m, len_n, len_r,
                         A, stride_A_m, stride_A_n,
                         U, stride_U_m, stride_U_r,
                         G, stride_G_n, stride_G_r);
    }
    else
    {
        cp_gradient_naive(comm, cfg, len_m, len_n, len_r,
                          A, stride_A_m, stride_A_n,
                          U, stride_U_m, stride_U_r,
                          G, stride_G_n, stride_G_r);
    }
}

#define FOREACH_TYPE(T) \
template void cp_gradient(const communicator& comm, const config& cfg, \
                          const len_vector& len_m, len_type len_n, len_type len_r, \
                          const T* A, \
                          const stride_vector& stride_A_m, stride_type stride_A_n, \
                          const ptr_vector<const T>& U, \
                          const stride_vector& stride_U_m, \
                          const stride_vector& stride_U_r, \
                          T* G, stride_type stride_G_n, stride_type stride_G_r);
#include "configs/foreach_type.h"

}
}
