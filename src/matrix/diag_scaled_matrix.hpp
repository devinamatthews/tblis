#ifndef _TBLIS_DIAG_SCALED_MATRIX_HPP_
#define _TBLIS_DIAG_SCALED_MATRIX_HPP_

#include "util/basic_types.h"

#include "normal_matrix.hpp"

namespace tblis
{

template <typename T>
class diag_scaled_matrix : public normal_matrix<T>
{
    protected:
        using normal_matrix<T>::data_;
        using normal_matrix<T>::tot_len_;
        using normal_matrix<T>::cur_len_;
        using normal_matrix<T>::off_;
        using normal_matrix<T>::stride_;
        unsigned diag_dim_ = 0;
        T* diag_ = nullptr;
        stride_type diag_stride_ = 0;

    public:
        diag_scaled_matrix() {}

        diag_scaled_matrix(len_type m, len_type n, T* ptr, stride_type rs, stride_type cs,
                           unsigned diag_dim, T* diag, stride_type inc)
        : normal_matrix<T>(m, n, ptr, rs, cs),
          diag_dim_(diag_dim), diag_(diag), diag_stride_(inc) {}

        T* diag() const
        {
            return diag_ + off_[diag_dim_]*diag_stride_;
        }

        stride_type diag_stride() const
        {
            return diag_stride_;
        }

        void transpose()
        {
            normal_matrix<T>::transpose();
            diag_dim_ = 1-diag_dim_;
        }

        void pack(const communicator& comm, const config& cfg, bool trans, normal_matrix<T>& Ap) const
        {
            const len_type MR = (!trans ? cfg.gemm_mr.def<T>()
                                        : cfg.gemm_nr.def<T>());
            const len_type ME = (!trans ? cfg.gemm_mr.extent<T>()
                                        : cfg.gemm_nr.extent<T>());
            const len_type KR = cfg.gemm_kr.def<T>();

            const len_type m_a = cur_len_[ trans];
            const len_type k_a = cur_len_[!trans];
            const stride_type rs_a = stride_[ trans];
            const stride_type cs_a = stride_[!trans];
            const stride_type inc_d = diag_stride_;

            TBLIS_ASSERT(diag_dim_ == !trans);

            comm.distribute_over_threads({m_a, MR}, {k_a, KR},
            [&](len_type m_first, len_type m_last, len_type k_first, len_type k_last)
            {
                const T*  p_a = this->data() +  m_first       *rs_a + k_first* cs_a;
                const T*  p_d = this->diag() +                        k_first*inc_d;
                      T* p_ap =    Ap.data() + (m_first/MR)*ME* k_a + k_first*   ME;

                for (len_type off_m = m_first;off_m < m_last;off_m += MR)
                {
                    len_type m = std::min(MR, m_last-off_m);
                    len_type k = k_last-k_first;

                    TBLIS_ASSERT(p_a + (m-1)*rs_a + (k-1)*cs_a <=
                                 data_ + (tot_len_[0]-1)*stride_[0] + (tot_len_[1]-1)*stride_[1]);
                    TBLIS_ASSERT(p_d + (k-1)*inc_d <= diag_ + (tot_len_[diag_dim_]-1)*stride_[diag_dim_]);
                    TBLIS_ASSERT(p_ap + k*ME <= Ap.data() + Ap.length(0)*Ap.length(1));

                    if (!trans)
                        cfg.pack_nnd_mr_ukr.call<T>(m, k, p_a, rs_a, cs_a, p_d, inc_d, p_ap);
                    else
                        cfg.pack_nnd_nr_ukr.call<T>(m, k, p_a, rs_a, cs_a, p_d, inc_d, p_ap);

                    p_a += m*rs_a;
                    p_ap += ME*k_a;
                }
            });
        }
};

}

#endif
