#ifndef _TBLIS_SCATTER_MATRIX_HPP_
#define _TBLIS_SCATTER_MATRIX_HPP_

#include "util/basic_types.h"
#include "util/macros.h"
#include "util/thread.h"

#include "memory/alignment.hpp"

#include "normal_matrix.hpp"

namespace tblis
{

template <typename T>
class scatter_matrix : public abstract_matrix<T>
{
    public:
        typedef const stride_type* scatter_type;

    protected:
        using abstract_matrix<T>::tot_len_;
        using abstract_matrix<T>::cur_len_;
        using abstract_matrix<T>::off_;
        T* data_ = nullptr;
        std::array<stride_type*, 2> scatter_ = {};

    public:
        scatter_matrix() {}

        scatter_matrix(T* A, row_view<stride_type> rscat,
                       row_view<stride_type> cscat)
        : data_(A), scatter_{rscat.data(), cscat.data()}
        {
            tot_len_ = cur_len_ = {rscat.length(0), cscat.length(0)};
        }

        void transpose()
        {
            using std::swap;
            abstract_matrix<T>::transpose();
            swap(scatter_[0], scatter_[1]);
        }

        void pack(const communicator& comm, const config& cfg,
                  bool trans, normal_matrix<T>& Ap) const
        {
            const len_type MR = (!trans ? cfg.gemm_mr.def<T>()
                                        : cfg.gemm_nr.def<T>());
            const len_type ME = (!trans ? cfg.gemm_mr.extent<T>()
                                        : cfg.gemm_nr.extent<T>());

            const len_type m_a = cur_len_[ trans];
            const len_type k_a = cur_len_[!trans];

            comm.distribute_over_threads({m_a, MR}, k_a,
            [&](len_type m_first, len_type m_last, len_type k_first, len_type k_last)
            {
                T* p_ap = Ap.data() + (m_first/MR)*ME*Ap.stride(trans) + k_first*ME;
                auto rscat_a = scatter_[ trans] + off_[ trans] + m_first;
                auto cscat_a = scatter_[!trans] + off_[!trans] + k_first;

                for (len_type m_off = m_first;m_off < m_last;)
                {
                    TBLIS_ASSERT(rscat_a - scatter_[ trans] < tot_len_[ trans]);
                    TBLIS_ASSERT(cscat_a - scatter_[!trans] < tot_len_[!trans]);

                    len_type m = std::min(MR, m_last - m_off);
                    len_type k = k_last - k_first;
                    const T* p_a = data_;

                    TBLIS_ASSERT(p_ap + k*ME <= Ap.data() + ceil_div(Ap.length(trans), MR)*ME*Ap.length(!trans));

                    if (!trans)
                        cfg.pack_ss_mr_ukr.call<T>(m, k, p_a, rscat_a, cscat_a, p_ap);
                    else
                        cfg.pack_ss_nr_ukr.call<T>(m, k, p_a, rscat_a, cscat_a, p_ap);

                    p_ap += ME*Ap.stride(trans);
                    m_off += MR;
                    rscat_a += MR;
                }
            });
        }
};

}

#endif
