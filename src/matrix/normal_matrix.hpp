#ifndef _TBLIS_NORMAL_MATRIX_HPP_
#define _TBLIS_NORMAL_MATRIX_HPP_

#include "abstract_matrix.hpp"

namespace tblis
{

template <typename T>
class normal_matrix : public abstract_matrix<T>
{
    template <typename> friend class patch_block_scatter_matrix;

    public:
        typedef T value_type;

    protected:
        using abstract_matrix<T>::data_;
        using abstract_matrix<T>::tot_len_;
        using abstract_matrix<T>::cur_len_;
        using abstract_matrix<T>::off_;
        std::array<stride_type,2> stride_ = {};

    public:
        normal_matrix() {}

        normal_matrix(len_type m, len_type n, T* ptr, stride_type rs, stride_type cs)
        : abstract_matrix<T>(m, n, ptr), stride_{rs, cs} {}

        T* data() const
        {
            return data_ + off_[0]*stride_[0] + off_[1]*stride_[1];
        }

        stride_type stride(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return stride_[dim];
        }

        const std::array<stride_type, 2>& strides() const
        {
            return stride_;
        }

        void transpose()
        {
            using std::swap;
            abstract_matrix<T>::transpose();
            swap(stride_[0], stride_[1]);
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

            comm.distribute_over_threads({m_a, MR}, {k_a, KR},
            [&](len_type m_first, len_type m_last, len_type k_first, len_type k_last)
            {
                const T*  p_a = this->data() +  m_first       *rs_a + k_first* cs_a;
                      T* p_ap =    Ap.data() + (m_first/MR)*ME* k_a + k_first*   ME;

                for (len_type off_m = m_first;off_m < m_last;off_m += MR)
                {
                    len_type m = std::min(MR, m_last-off_m);
                    len_type k = k_last-k_first;

                    TBLIS_ASSERT(p_a + (m-1)*rs_a + (k-1)*cs_a <=
                                 data_ + (tot_len_[0]-1)*stride_[0] + (tot_len_[1]-1)*stride_[1]);
                    TBLIS_ASSERT(p_ap + k*ME <= Ap.data() + Ap.length(0)*Ap.length(1));

                    if (!trans)
                        cfg.pack_nn_mr_ukr.call<T>(m, k, p_a, rs_a, cs_a, p_ap);
                    else
                        cfg.pack_nn_nr_ukr.call<T>(m, k, p_a, rs_a, cs_a, p_ap);

                    p_a += m*rs_a;
                    p_ap += ME*k_a;
                }
            });
        }
};

}

#endif
