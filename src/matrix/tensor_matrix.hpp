#ifndef _TBLIS_TENSOR_MATRIX_HPP_
#define _TBLIS_TENSOR_MATRIX_HPP_

#include "util/basic_types.h"

#include "external/stl_ext/include/algorithm.hpp"

#include "abstract_matrix.hpp"
#include "packed_matrix.hpp"

namespace tblis
{

struct tensor_matrix_impl
{
    char* data_ = nullptr;
    std::array<len_vector, 2> lens_ = {};
    std::array<stride_vector, 2> strides_ = {};
    std::array<bool, 2> pack_3d_ = {};

    tensor_matrix_impl(char* ptr,
                       const len_vector& len_m,
                       const len_vector& len_n,
                       const stride_vector& stride_m,
                       const stride_vector& stride_n,
                       bool pack_m_3d,
                       bool pack_n_3d)
    : data_(ptr), lens_{len_m, len_n}, strides_{stride_m, stride_n},
      pack_3d_{pack_m_3d, pack_n_3d} {}
};

template <typename T>
struct is_tensor_helper : std::false_type {};

template <typename T, int N, typename Allocator>
struct is_tensor_helper<marray<T,N,Allocator>> : std::true_type {};

template <typename T, int N>
struct is_tensor_helper<marray_view<T,N>> : std::true_type {};

template <typename T>
struct is_tensor : is_tensor_helper<typename std::decay<T>::type> {};

class tensor_matrix : public abstract_matrix_adapter<tensor_matrix,tensor_matrix_impl>
{
    protected:

        static abstract_matrix do_pack(abstract_matrix& A_,
                                       const communicator& comm, const config& cfg,
                                       int mat, MemoryPool& pool)
        {
            auto& A = static_cast<tensor_matrix&>(A_);
            const type_t type = A.type();
            const bool trans = mat == matrix_constants::MAT_B;
            const len_type MR = (!trans ? cfg.gemm_mr.def(type)
                                        : cfg.gemm_nr.def(type));
            const len_type ME = (!trans ? cfg.gemm_mr.extent(type)
                                        : cfg.gemm_nr.extent(type));
            const len_type KR = cfg.gemm_kr.def(type);
            const len_type KE = cfg.gemm_kr.extent(type);

            const stride_type ts = type_size[type];
            const len_type m = A.length( trans);
            const len_type k = A.length(!trans);

            const len_type m_p = ceil_div(m, MR)*ME;
            const len_type k_p = round_up(k, KE);
            const len_type MC = std::max(!trans ? cfg.gemm_mc.max(type)
                                                : cfg.gemm_nc.max(type), m_p);
            const len_type KC = std::max(cfg.gemm_kc.max(type), k_p);
            const stride_type nelem = MC*KC + std::max(MC,KC)*TBLIS_MAX_UNROLL +
                                      2*size_as_type<stride_type>(MC+KC, type);

            packed_matrix P(type, !trans ? m : k_p, !trans ? k_p : m,
                            A.get_buffer(comm, nelem, pool), k_p*ME);

            stride_type* rscat = convert_and_align<stride_type>(P.data() + m_p*k_p*ts);
            stride_type* cscat = rscat + m_p;
            stride_type* rbs = cscat + k_p;
            stride_type* cbs = rbs + m_p;

            A.fill_block_scatter(comm, trans, MR, KR, rscat, cscat, rbs, cbs);

            pack(comm, cfg, A.scale(), A.conj(), trans, m, k, A.data(),
                 rscat, cscat, rbs, cbs, P.data(), P.panel_stride());

            return P;
        }

        static void do_gemm(abstract_matrix& C_, const communicator& comm, const config& cfg,
                            MemoryPool& pool, const abstract_matrix& A_, const abstract_matrix& B_)
        {
            auto& A = static_cast<const packed_matrix&>(A_);
            auto& B = static_cast<const packed_matrix&>(B_);
            auto& C = static_cast<      tensor_matrix&>(C_);

            const type_t type = C.type();
            const len_type MR = cfg.gemm_mr.def(type);
            const len_type NR = cfg.gemm_nr.def(type);
            const len_type m = C.length(0);
            const len_type n = C.length(1);
            const len_type k = A.length(1);
            const len_type MC = std::max(cfg.gemm_mc.max(type), m);
            const len_type NC = std::max(cfg.gemm_nc.max(type), n);
            const stride_type nelem = 2*size_as_type<stride_type>(MC+NC, type);

            stride_type* rscat = convert_and_align<stride_type>(C.get_buffer(comm, nelem, pool));
            stride_type* cscat = rscat + m;
            stride_type* rbs = cscat + n;
            stride_type* cbs = rbs + m;

            C.fill_block_scatter(comm, false, MR, NR, rscat, cscat, rbs, cbs);

            gemm(comm, cfg, C.scale(), C.conj(), m, n, k,
                 A.data(), A.panel_stride(), B.data(), B.panel_stride(),
                 C.data(), rscat, cscat, rbs,  cbs);
        }

    public:
        tensor_matrix(const scalar& alpha, bool conj,
                      const len_vector& len_m,
                      const len_vector& len_n,
                      char* ptr,
                      const stride_vector& stride_m,
                      const stride_vector& stride_n,
                      bool pack_m_3d = false,
                      bool pack_n_3d = false)
        : abstract_matrix_adapter(alpha, conj, 0, 0, false,
                                  ptr, len_m, len_n, stride_m, stride_n,
                                  pack_m_3d, pack_n_3d)
        {
            TBLIS_ASSERT(lengths(0).size() == strides(0).size());
            TBLIS_ASSERT(lengths(1).size() == strides(1).size());

            len_type m = stl_ext::prod(lengths(0));
            len_type n = stl_ext::prod(lengths(1));
            bool row_major = !strides(1).empty() &&
                             strides(1)[0] == 1 &&
                             n > 1;

            reset(row_major, m, n);

            pack_ = &do_pack;
            gemm_ = &do_gemm;
        }
        static void fill_scatter_3d(type_t type, len_type m0, len_vector len,
                                    len_type s0, stride_vector stride, len_type BS,
                                    stride_type off, stride_type size,
                                    stride_type* scat)
        {
            const len_type CL = 64/type_size[type];

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

        static void fill_scatter(type_t type, len_vector len, stride_vector stride, len_type BS,
                                 stride_type off, stride_type size,
                                 stride_type* scat, bool pack_3d)
        {
            if (size == 0) return;

            if (len.empty())
            {
                *scat = 0;
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

            if (pack_3d)
            {
                fill_scatter_3d(type, m0, len, s0, stride, BS, off, size, scat);
            }
            else
            {
                viterator<> it(len, stride);

                len_type off0, p0;
                detail::divide(off, m0, off0, p0);
                stride_type pos = 0;
                it.position(off0, pos);

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
        }

        static void fill_block_stride(len_type BS, stride_type size,
                                      stride_type* scat, stride_type* bs)
        {
            if (size == 0) return;

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

        static void fill_block_scatter(type_t type, len_vector len, stride_vector stride, len_type BS,
                                       stride_type off, stride_type size,
                                       stride_type* scat, stride_type* bs,
                                       bool pack_3d)
        {
            if (size == 0) return;

            fill_scatter(type, len, stride, BS, off, size, scat, pack_3d);
            fill_block_stride(BS, size, scat, bs);
        }

        void fill_block_scatter(const communicator& comm, bool trans,
                                len_type MR, len_type NR,
                                stride_type* rscat, stride_type* cscat,
                                stride_type* rbs, stride_type* cbs)
        {
            if (comm.master())
            {
                fill_block_scatter(type(), lengths( trans), strides( trans), MR,
                                   offset( trans), length( trans), rscat, rbs,
                                   pack_3d( trans));

                fill_block_scatter(type(), lengths(!trans), strides(!trans), NR,
                                   offset(!trans), length(!trans), cscat, cbs,
                                   pack_3d(!trans));
            }

            comm.barrier();
        }

        static void pack(const communicator& comm, const config& cfg,
                         const scalar& alpha, bool conj, bool trans,
                         len_type m, len_type k, char* p_a,
                         const stride_type* rscat, const stride_type* cscat,
                         const stride_type* rbs, const stride_type* cbs,
                         char* p_ap, stride_type ps_ap)
        {
            const type_t type = alpha.type;
            const len_type MR = (!trans ? cfg.gemm_mr.def(type)
                                        : cfg.gemm_nr.def(type));
            const len_type ME = (!trans ? cfg.gemm_mr.extent(type)
                                        : cfg.gemm_nr.extent(type));
            const len_type KR = cfg.gemm_kr.def(type);
            const stride_type ts = type_size[type];

            comm.distribute_over_threads({m, MR}, {k, KR},
            [&](len_type m_first, len_type m_last, len_type k_first, len_type k_last)
            {
                char* p_ap1 = p_ap + (m_first/MR)*ps_ap*ts + k_first*ME*ts;

                for (len_type m_off = m_first;m_off < m_last;m_off += MR)
                {
                    const stride_type rs_a = rbs ? rbs[m_off] : 0;
                    const len_type m_loc = std::min(MR, m-m_off);
                    const len_type k_loc = k_last-k_first;

                    TBLIS_ASSERT(p_ap1 >= p_ap);
                    TBLIS_ASSERT(p_ap1+k_loc*ME*ts <= p_ap+ceil_div(m,MR)*ps_ap*ts);

                    if (rs_a)
                    {
                        if (!trans)
                            cfg.pack_nb_mr_ukr.call(type, m_loc, k_loc,
                                                    &alpha, conj, p_a + rscat[m_off]*ts,
                                                    rs_a, cscat+k_first, cbs+k_first, p_ap1);
                        else
                            cfg.pack_nb_nr_ukr.call(type, m_loc, k_loc,
                                                    &alpha, conj, p_a + rscat[m_off]*ts,
                                                    rs_a, cscat+k_first, cbs+k_first, p_ap1);
                    }
                    else
                    {
                        if (!trans)
                            cfg.pack_ss_mr_ukr.call(type, m_loc, k_loc,
                                                    &alpha, conj, p_a,
                                                    rscat+m_off, cscat+k_first, p_ap1);
                        else
                            cfg.pack_ss_nr_ukr.call(type, m_loc, k_loc,
                                                    &alpha, conj, p_a,
                                                    rscat+m_off, cscat+k_first, p_ap1);
                    }

                    p_ap1 += ps_ap*ts;
                }
            });
        }

        using abstract_matrix::pack;

        static void gemm(const communicator& comm, const config& cfg,
                         const scalar& beta, bool conj,
                         len_type m, len_type n, len_type k,
                         char* p_a, stride_type ps_a,
                         char* p_b, stride_type ps_b, char* p_c,
                         const stride_type* rscat, const stride_type* cscat,
                         const stride_type* rbs, const stride_type* cbs)
        {
            const type_t type = beta.type;
            const len_type MR = cfg.gemm_mr.def(type);
            const len_type NR = cfg.gemm_nr.def(type);
            const bool row_major = cfg.gemm_row_major.value(type);
            const stride_type ts = type_size[type];
            const len_type m_first = 0;
            const len_type m_last = ceil_div(m, MR);
            const stride_type rs_ab = row_major ? NR : 1;
            const stride_type cs_ab = row_major ? 1 : MR;

            comm.distribute_over_threads(ceil_div(n, NR),
            [&](len_type n_first, len_type n_last)
            {
                constexpr static dcomplex zero{};
                char p_ab[8192] __attribute__((aligned(64)));

                for (len_type n_off = n_first;n_off < n_last;n_off++)
                for (len_type m_off = m_first;m_off < m_last;m_off++)
                {
                    len_type m_loc = std::min(MR, m-m_off*MR);
                    len_type n_loc = std::min(NR, n-n_off*NR);

                    stride_type rs_c = rbs ? rbs[m_off*MR] : 0;
                    stride_type cs_c = cbs ? cbs[n_off*NR] : 0;

                    char* p_a1 = p_a + m_off*ps_a*ts;
                    char* p_b1 = p_b + n_off*ps_b*ts;
                    char* p_c1 = p_c + rscat[m_off*MR]*ts + cscat[n_off*NR]*ts;

                    auxinfo_t aux{p_a1, p_b1, p_c1};

                    if (rs_c && cs_c)
                    {
                        cfg.gemm_ukr.call(type, m_loc, n_loc, k, p_a1, p_b1,
                                          &beta, p_c1, rs_c, cs_c, &aux);
                    }
                    else
                    {
                        cfg.gemm_ukr.call(type, MR, NR, k, p_a1, p_b1,
                                          &zero, p_ab, rs_ab, cs_ab, &aux);

                        cfg.update_ss_ukr.call(type, m_loc, n_loc, p_ab,
                                               &beta, p_c, // NOT p_c1
                                               rscat + m_off*MR,
                                               cscat + n_off*NR);
                    }
                }
            });
        }

        using abstract_matrix::gemm;

        char* data() const
        {
            return impl().data_;
        }

        const len_vector& lengths(int dim) const
        {
            TBLIS_ASSERT(dim >= 0 && dim < 2);
            return impl().lens_[dim^transposed()];
        }

        const stride_vector& strides(int dim) const
        {
            TBLIS_ASSERT(dim >= 0 && dim < 2);
            return impl().strides_[dim^transposed()];
        }

        bool pack_3d(int dim) const
        {
            TBLIS_ASSERT(dim >= 0 && dim < 2);
            return impl().pack_3d_[dim^transposed()];
        }
};

}

#endif
