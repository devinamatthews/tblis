#ifndef _TBLIS_BLOCK_SCATTER_MATRIX_HPP_
#define _TBLIS_BLOCK_SCATTER_MATRIX_HPP_

#include "util/basic_types.h"
#include "util/macros.h"
#include "util/thread.h"

#include "memory/alignment.hpp"

#include "normal_matrix.hpp"
#include "tensor_matrix.hpp"
#include "dpd_tensor_matrix.hpp"

namespace tblis
{

inline len_type gcd(len_type a, len_type b)
{
    a = std::abs(a);
    b = std::abs(b);

    if (a == 0) return b;
    if (b == 0) return a;

    unsigned d = __builtin_ctzl(a|b);

    a >>= __builtin_ctzl(a);
    b >>= __builtin_ctzl(b);

    while (a != b)
    {
        if (a > b)
        {
            a = (a-b)>>1;
        }
        else
        {
            b = (b-a)>>1;
        }
    }

    return a<<d;
}

inline len_type lcm(len_type a, len_type b)
{
    return a*b/gcd(a,b);
}

template <typename T>
class block_scatter_matrix : public abstract_matrix<T>
{
    template <typename> friend class patch_block_scatter_matrix;

    public:
        typedef const stride_type* scatter_type;

    protected:
        using abstract_matrix<T>::data_;
        using abstract_matrix<T>::tot_len_;
        using abstract_matrix<T>::cur_len_;
        using abstract_matrix<T>::off_;
        std::array<stride_type*, 2> scatter_ = {};
        std::array<stride_type*, 2> block_stride_ = {};
        std::array<len_type, 2> block_size_ = {};

        static void fill_block_scatter(len_vector len, stride_vector stride, len_type BS,
                                       stride_type off, stride_type size,
                                       stride_type* scat, stride_type* bs,
                                       bool pack_3d = false)
        {
            if (size == 0) return;

            constexpr len_type CL = 64 / sizeof(T);

            if (len.empty())
            {
                *scat = 0;
                *bs = 1;
                return;
            }

            stride_type tot_len = 1;
            for (auto& l : len) tot_len *= l;
            TBLIS_ASSERT(off >= 0);
            TBLIS_ASSERT(size >= 0);
            TBLIS_ASSERT(off+size <= tot_len);

            len_type m0 = len[0];
            stride_type s0 = stride[0];

            len.erase(len.begin());
            stride.erase(stride.begin());

            if (!pack_3d)
            {
                viterator<> it(len, stride);

                len_type p0 = off%m0;
                stride_type pos = 0;
                it.position(off/m0, pos);

                for (len_type idx = 0;idx < size && it.next(pos);)
                {
                    auto pos2 = pos + p0*s0;
                    auto imax = std::min(m0-p0,size-idx);
                    for (len_type i = 0;i < imax;i++)
                    {
                        scat[idx++] = pos2;
                        pos2 += s0;
                    }
                    p0 = 0;
                }
            }
            else
            {
                TBLIS_ASSERT(!len.empty());

                len_type m1 = len[0];
                stride_type s1 = stride[0];

                len_type BS0 = std::min(lcm(BS, CL), std::max(BS, m0 - m0%BS));
                len_type BS1 = std::min(        CL , std::max(BS, m1 - m1%BS));

                len.erase(len.begin());
                stride.erase(stride.begin());

                viterator<> it(len, stride);

                len_type p01 = off%(m0*m1);
                len_type p0 = p01%m0;
                len_type p1 = p01/m0;
                len_type q01 = (off+size-1)%(m0*m1);
                len_type q0 = q01%m0;
                len_type q1 = q01/m0;
                len_type r0 = q0 - q0%BS0;
                stride_type pos = 0;
                len_type b_min = off/(m0*m1);
                len_type b_max = (off+size-1)/(m0*m1)+1;

                if (p0 == 0) p1--;
                if (q0 == m0-1) q1++;

                it.position(b_min, pos);
                auto it0 = it;
                auto pos0 = pos;

                /*
                 *                  p0  q0      r0
                 *    +---+---+---+---+---+---+---+
                 *    |   |   |   |   |   |   |   |
                 *    +---+---+---#===============#
                 * p1 |   |   |   #p01| C | C | C #
                 *    #=======#===#===#=======#===#
                 *    # A | A # A | A # A | A # B #
                 *    #---+---#---+---#---+---#---#
                 *    # A | A # A | A # A | A # B #
                 *    #=======#=======#=======#===#
                 *    # A'| A'# A'| A'# A'| A'# B'#
                 *    #=======#=======#===#===#===#
                 * q1 # D | D | D | D |q01#   |   |
                 *    #===================#---+---+
                 *    |   |   |   |   |   |   |   |
                 *    +---+---+---+---+---+---+---+
                 */

                /*
                 * A:  Full BS0*BS1 blocks
                 * A': Partial BS0*n blocks
                 */

                len_type idx = 0;

                if (r0 > 0)
                {
                    for (len_type b = b_min;b < b_max && it.next(pos);b++)
                    {
                        len_type min1 = b == b_min ? p1+1 : 0;
                        len_type max1 = b == b_max-1 ? q1 : m1;

                        for (len_type i1 = min1;i1 < max1;i1 += BS1)
                        for (len_type i0 = 0;i0 < r0;i0 += BS0)
                        for (len_type j1 = 0;j1 < std::min(BS1, max1-i1);j1++)
                        for (len_type j0 = 0;j0 < BS0;j0++)
                            scat[idx++] = pos + (i0+j0)*s0 + (i1+j1)*s1;
                    }

                    it = it0;
                    pos = pos0;
                }

                /*
                 * B:  Partial m*BS1 blocks
                 * B': Partial m*n block
                 * C:  First partial row
                 * D:  Last partial row
                 */

                for (len_type b = b_min;b < b_max && it.next(pos);b++)
                {
                    len_type min1 = b == b_min ? p1+1 : 0;
                    len_type max1 = b == b_max-1 ? q1 : m1;

                    for (len_type j1 = min1;j1 < max1;j1++)
                    for (len_type j0 = r0;j0 < m0;j0++)
                        scat[idx++] = pos + j0*s0 + j1*s1;

                    bool do_p = b == b_min && p0 > 0;
                    bool do_q = b == b_max-1 && q0 < m0-1;

                    if (do_p && do_q && p1 == q1)
                    {
                        // p01 and q01 are on the same row
                        for (len_type j0 = p0;j0 <= q0;j0++)
                            scat[idx++] = pos + j0*s0 + p1*s1;
                    }
                    else
                    {
                        if (do_p)
                            for (len_type j0 = p0;j0 < m0;j0++)
                                scat[idx++] = pos + j0*s0 + p1*s1;

                        if (do_q)
                            for (len_type j0 = 0;j0 <= q0;j0++)
                                scat[idx++] = pos + j0*s0 + q1*s1;
                    }
                }

                TBLIS_ASSERT(idx == size);
            }

            for (len_type i = 0;i < size;i += BS)
            {
                len_type bl = std::min(size-i, BS);
                stride_type s = bl > 1 ? scat[i+1]-scat[i] : 1;
                for (len_type j = i+1;j+1 < i+bl;j++)
                {
                    if (scat[j+1]-scat[j] != s) s = 0;
                }
                bs[i] = s;
            }
        }

    public:
        block_scatter_matrix() {}

        block_scatter_matrix(const communicator& comm, const tensor_matrix<T>& A,
                             len_type MB, stride_type* rscat, stride_type* rbs,
                             len_type NB, stride_type* cscat, stride_type* cbs)
        {
            data_ = A.data_;
            tot_len_ = cur_len_ = A.cur_len_;
            scatter_ = {rscat, cscat};
            block_stride_ = {rbs, cbs};
            block_size_ = {MB, NB};

            if (comm.master())
            {
                fill_block_scatter(A.lens_[0], A.strides_[0], block_size_[0], A.off_[0],
                                   tot_len_[0], scatter_[0], block_stride_[0], A.pack_3d_[0]);
                fill_block_scatter(A.lens_[1], A.strides_[1], block_size_[1], A.off_[1],
                                   tot_len_[1], scatter_[1], block_stride_[1], A.pack_3d_[1]);
            }

            comm.barrier();
        }

        block_scatter_matrix(const communicator& comm, const dpd_tensor_matrix<T>& A,
                             len_type MB, stride_type* rscat, stride_type* rbs,
                             len_type NB, stride_type* cscat, stride_type* cbs)
        {
            abort();
        }

        len_type block_size(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return block_size_[dim];
        }

        void transpose()
        {
            using std::swap;
            abstract_matrix<T>::transpose();
            swap(scatter_[0], scatter_[1]);
            swap(block_stride_[0], block_stride_[1]);
            swap(block_size_[0], block_size_[1]);
        }

        void block(T*& data,
                   scatter_type& rscat, stride_type& rbs, len_type& rbl,
                   scatter_type& cscat, stride_type& cbs, len_type& cbl) const
        {
            rscat = scatter_[0] + off_[0];
            cscat = scatter_[1] + off_[1];

            TBLIS_ASSERT(off_[0]%block_size_[0] == 0);
            TBLIS_ASSERT(off_[1]%block_size_[1] == 0);

            rbs = block_stride_[0][off_[0]];
            cbs = block_stride_[1][off_[1]];

            rbl = std::min(block_size_[0], cur_len_[0]);
            cbl = std::min(block_size_[1], cur_len_[1]);

            data = data_ + (rbs ? *rscat : 0) + (cbs ? *cscat : 0);
        }

        void pack(const communicator& comm, const config& cfg,
                  bool trans, normal_matrix<T>& Ap) const
        {
            const len_type MR = (!trans ? cfg.gemm_mr.def<T>()
                                        : cfg.gemm_nr.def<T>());
            const len_type ME = (!trans ? cfg.gemm_mr.extent<T>()
                                        : cfg.gemm_nr.extent<T>());
            const len_type KR = cfg.gemm_kr.def<T>();

            TBLIS_ASSERT(block_size_[ trans] == MR);
            TBLIS_ASSERT(block_size_[!trans] == KR);

            const len_type m_a = cur_len_[ trans];
            const len_type k_a = cur_len_[!trans];

            comm.distribute_over_threads({m_a, MR}, {k_a, KR},
            [&](len_type m_first, len_type m_last, len_type k_first, len_type k_last)
            {
                T* p_ap = Ap.data() + (m_first/MR)*ME*Ap.stride(trans) + k_first*ME;
                scatter_type rscat_a = scatter_[ trans] + off_[ trans] + m_first;
                scatter_type cscat_a = scatter_[!trans] + off_[!trans] + k_first;
                scatter_type rbs_a = block_stride_[ trans] + off_[ trans] + m_first;
                scatter_type cbs_a = block_stride_[!trans] + off_[!trans] + k_first;

                for (len_type m_off = m_first;m_off < m_last;)
                {
                    TBLIS_ASSERT(rscat_a - scatter_[ trans] < tot_len_[ trans]);
                    TBLIS_ASSERT(cscat_a - scatter_[!trans] < tot_len_[!trans]);
                    TBLIS_ASSERT(rbs_a - block_stride_[ trans] < tot_len_[ trans]);
                    TBLIS_ASSERT(cbs_a - block_stride_[!trans] < tot_len_[!trans]);

                    len_type m = std::min(MR, m_last - m_off);
                    len_type k = k_last - k_first;
                    stride_type rs_a = *rbs_a;
                    const T* p_a = data_ + (rs_a ? *rscat_a : 0);

                    TBLIS_ASSERT(p_ap + k*ME <= Ap.data() + ceil_div(Ap.length(trans), MR)*ME*Ap.length(!trans));

                    if (rs_a)
                    {
                        if (!trans)
                            cfg.pack_nb_mr_ukr.call<T>(m, k, p_a, rs_a, cscat_a, cbs_a, p_ap);
                        else
                            cfg.pack_nb_nr_ukr.call<T>(m, k, p_a, rs_a, cscat_a, cbs_a, p_ap);
                    }
                    else
                    {
                        if (!trans)
                            cfg.pack_sb_mr_ukr.call<T>(m, k, p_a, rscat_a, cscat_a, cbs_a, p_ap);
                        else
                            cfg.pack_sb_nr_ukr.call<T>(m, k, p_a, rscat_a, cscat_a, cbs_a, p_ap);
                    }

                    p_ap += ME*Ap.stride(trans);
                    m_off += MR;
                    rscat_a += MR;
                    rbs_a += MR;
                }
            });
        }
};

}

#endif
