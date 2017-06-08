#ifndef _TBLIS_BLOCK_SCATTER_MATRIX_HPP_
#define _TBLIS_BLOCK_SCATTER_MATRIX_HPP_

#include "util/basic_types.h"
#include "util/macros.h"
#include "util/thread.h"

#include "tensor_matrix.hpp"

namespace tblis
{

template <typename T>
class block_scatter_matrix
{
    public:
        typedef size_t size_type;
        typedef const stride_type* scatter_type;
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;

    protected:
        pointer data_;
        std::array<len_type, 2> len_;
        std::array<scatter_type, 2> block_scatter_;
        std::array<scatter_type, 2> scatter_;
        std::array<len_type, 2> block_size_;

    public:
        block_scatter_matrix()
        {
            reset();
        }

        block_scatter_matrix(const block_scatter_matrix&) = default;

        block_scatter_matrix(len_type m, len_type n, pointer p,
                             scatter_type rscat, len_type MB, scatter_type rbs,
                             scatter_type cscat, len_type NB, scatter_type cbs)
        {
            reset(m, n, p, rscat, MB, rbs, cscat, NB, cbs);
        }

        block_scatter_matrix& operator=(const block_scatter_matrix&) = delete;

        void reset()
        {
            data_ = nullptr;
            len_[0] = 0;
            len_[1] = 0;
            block_scatter_[0] = nullptr;
            block_scatter_[1] = nullptr;
            scatter_[0] = nullptr;
            scatter_[1] = nullptr;
        }

        void reset(const block_scatter_matrix& other)
        {
            data_ = other.data_;
            len_[0] = other.len_[0];
            len_[1] = other.len_[1];
            block_scatter_[0] = other.block_scatter_[0];
            block_scatter_[1] = other.block_scatter_[1];
            scatter_[0] = other.scatter_[0];
            scatter_[1] = other.scatter_[1];
        }

        void reset(len_type m, len_type n, pointer p,
                   scatter_type rscat, len_type MB, scatter_type rbs,
                   scatter_type cscat, len_type NB, scatter_type cbs)
        {
            data_ = p;
            len_[0] = m;
            len_[1] = n;
            block_scatter_[0] = rbs;
            block_scatter_[1] = cbs;
            scatter_[0] = rscat;
            scatter_[1] = cscat;
            block_size_[0] = MB;
            block_size_[1] = NB;

            for (len_type i = 0;i < m;i += MB)
            {
                stride_type s = (m-i) > 1 ? rscat[i+1]-rscat[i] : 1;
                for (len_type j = i+1;j+1 < std::min(i+MB,m);j++)
                {
                    if (rscat[j+1]-rscat[j] != s) s = 0;
                }
                TBLIS_ASSERT(s == -1 || s == rbs[i/MB]);
            }

            for (len_type i = 0;i < n;i += NB)
            {
                stride_type s = (n-i) > 1 ? cscat[i+1]-cscat[i] : 1;
                for (len_type j = i+1;j+1 < std::min(i+NB,n);j++)
                {
                    if (cscat[j+1]-cscat[j] != s) s = 0;
                }
                TBLIS_ASSERT(s == -1 || s == cbs[i/NB]);
            }
        }

        len_type block_size(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return block_size_[dim];
        }

        len_type length(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return len_[dim];
        }

        len_type length(unsigned dim, len_type m)
        {
            TBLIS_ASSERT(dim < 2);
            std::swap(m, len_[dim]);
            return m;
        }

        stride_type stride(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return *block_scatter_[dim];
        }

        scatter_type scatter(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return scatter_[dim];
        }

        scatter_type block_scatter(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return block_scatter_[dim];
        }

        void shift(unsigned dim, len_type n)
        {
            TBLIS_ASSERT(dim < 2);
            scatter_[dim] += n;
            block_scatter_[dim] += ceil_div(n, block_size_[dim]);
        }

        void shift_down(unsigned dim)
        {
            shift(dim, length(dim));
        }

        void shift_up(unsigned dim)
        {
            shift(dim, -length(dim));
        }

        void shift_block(unsigned dim, len_type n)
        {
            TBLIS_ASSERT(dim < 2);
            scatter_[dim] += n*block_size_[dim];
            block_scatter_[dim] += n;
        }

        pointer data()
        {
            return data_ + (stride(0) == 0 ? 0 : *scatter_[0])
                         + (stride(1) == 0 ? 0 : *scatter_[1]);
        }

        const_pointer data() const
        {
            return const_cast<block_scatter_matrix&>(*this).data();
        }

        pointer raw_data() { return data_; }

        const_pointer raw_data() const { return data_; }
};

template <typename T>
void add(const communicator& comm, T alpha, tensor_matrix<T> A,
                                   T  beta,   matrix_view<T> B)
{
    constexpr len_type MB = 4;
    constexpr len_type NB = 4;

    TBLIS_ASSERT(A.length(0) == B.length(0));
    TBLIS_ASSERT(A.length(1) == B.length(1));

    len_type m = A.length(0);
    len_type n = A.length(1);
    stride_type rs0_A = A.stride(0);
    stride_type cs0_A = A.stride(1);
    stride_type rs_B = B.stride(0);
    stride_type cs_B = B.stride(1);

    std::vector<stride_type> rscat_A(m);
    std::vector<stride_type> cscat_A(n);
    std::vector<stride_type> rbs_A((m+MB-1)/MB);
    std::vector<stride_type> cbs_A((n+NB-1)/NB);

    A.fill_block_scatter(0, rscat_A.data(), MB, rbs_A.data());
    A.fill_block_scatter(1, cscat_A.data(), NB, cbs_A.data());

    stride_type rs_min = std::min(rs0_A, rs_B);
    stride_type rs_max = std::max(rs0_A, rs_B);
    stride_type cs_min = std::min(cs0_A, cs_B);
    stride_type cs_max = std::max(cs0_A, cs_B);

    len_type m_min, m_max, n_min, n_max;
    std::tie(m_min, m_max, std::ignore,
             n_min, n_max, std::ignore) =
        comm.distribute_over_threads_2d(m, n, MB, NB);

    if (rs_min == 1 && (cs_min != 1 || cs_max < rs_max))
    {
        for (len_type j0 = n_min, jb = n_min/NB;j0 < n_max;j0 += NB, jb++)
        {
            len_type n_loc = std::min(NB, n_max-j0);

            for (len_type i0 = m_min, ib = m_min/MB;i0 < m_max;i0 += MB, ib++)
            {
                len_type m_loc = std::min(MB, m_max-i0);

                stride_type rs_A = rbs_A[ib];
                const T* p_A = A.data() + (rs_A ? rscat_A[i0] : 0);
                T* p_B = B.data() + rs_B*i0 + cs_B*j0;

                if (rs_A)
                {
                    TBLIS_SPECIAL_CASE(m_loc == MB,
                    TBLIS_SPECIAL_CASE(n_loc == NB,
                    TBLIS_SPECIAL_CASE(rs_A == 1,
                    TBLIS_SPECIAL_CASE(rs_B == 1,
                    TBLIS_SPECIAL_CASE(alpha == T(1),
                    TBLIS_SPECIAL_CASE(beta == T(0),
                    for (len_type j = 0;j < n_loc;j++)
                    {
                        for (len_type i = 0;i < m_loc;i++)
                        {
                            p_B[i*rs_B + j*cs_B] =
                                alpha*p_A[i*rs_A + cscat_A[j0+j]] +
                                beta*p_B[i*rs_B + j*cs_B];
                        }
                    }
                    ))))));
                }
                else
                {
                    for (len_type j = 0;j < n_loc;j++)
                    {
                        for (len_type i = 0;i < m_loc;i++)
                        {
                            p_B[i*rs_B + j*cs_B] =
                                alpha*p_A[rscat_A[i0+i] + cscat_A[j0+j]] +
                                beta*p_B[i*rs_B + j*cs_B];
                        }
                    }
                }
            }
        }
    }
    else
    {
        for (len_type i0 = m_min, ib = m_min/MB;i0 < m_max;i0 += MB, ib++)
        {
            len_type m_loc = std::min(MB, m_max-i0);

            for (len_type j0 = n_min, jb = n_min/NB;j0 < n_max;j0 += NB, jb++)
            {
                len_type n_loc = std::min(NB, n_max-j0);

                stride_type cs_A = cbs_A[jb];
                const T* p_A = A.data() + (cs_A ? cscat_A[j0] : 0);
                T* p_B = B.data() + rs_B*i0 + cs_B*j0;

                if (cs_A)
                {
                    TBLIS_SPECIAL_CASE(m_loc == MB,
                    TBLIS_SPECIAL_CASE(n_loc == NB,
                    TBLIS_SPECIAL_CASE(cs_A == 1,
                    TBLIS_SPECIAL_CASE(cs_B == 1,
                    TBLIS_SPECIAL_CASE(alpha == T(1),
                    TBLIS_SPECIAL_CASE(beta == T(0),
                    for (len_type i = 0;i < m_loc;i++)
                    {
                        for (len_type j = 0;j < n_loc;j++)
                        {
                            p_B[i*rs_B + j*cs_B] =
                                alpha*p_A[rscat_A[i0+i] + j*cs_A] +
                                beta*p_B[i*rs_B + j*cs_B];
                        }
                    }
                    ))))));
                }
                else
                {
                    for (len_type j = 0;j < n_loc;j++)
                    {
                        for (len_type i = 0;i < m_loc;i++)
                        {
                            p_B[i*rs_B + j*cs_B] =
                                alpha*p_A[rscat_A[i0+i] + cscat_A[j0+j]] +
                                beta*p_B[i*rs_B + j*cs_B];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void add(T alpha, tensor_matrix<T> A, T beta, matrix_view<T> B)
{
    parallelize_if(
        [&](const communicator& comm)
        {
            add(comm, alpha, A, beta, B);
        },
        nullptr);
}

template <typename T>
void add(single_t, T alpha, tensor_matrix<T> A, T beta, matrix_view<T> B)
{
    communicator comm;
    add(comm, alpha, A, beta, B);
}

template <typename T>
void add(const communicator& comm, T alpha, matrix_view<const T> A,
                                   T  beta,     tensor_matrix<T> B)
{
    constexpr len_type MB = 4;
    constexpr len_type NB = 4;

    TBLIS_ASSERT(A.length(0) == B.length(0));
    TBLIS_ASSERT(A.length(1) == B.length(1));

    len_type m = B.length(0);
    len_type n = B.length(1);
    stride_type rs0_B = B.stride(0);
    stride_type cs0_B = B.stride(1);
    stride_type rs_A = A.stride(0);
    stride_type cs_A = A.stride(1);

    std::vector<stride_type> rscat_B(m);
    std::vector<stride_type> cscat_B(n);
    std::vector<stride_type> rbs_B((m+MB-1)/MB);
    std::vector<stride_type> cbs_B((n+NB-1)/NB);

    B.fill_block_scatter(0, rscat_B.data(), MB, rbs_B.data());
    B.fill_block_scatter(1, cscat_B.data(), NB, cbs_B.data());

    stride_type rs_min = std::min(rs0_B, rs_A);
    stride_type rs_max = std::max(rs0_B, rs_A);
    stride_type cs_min = std::min(cs0_B, cs_A);
    stride_type cs_max = std::max(cs0_B, cs_A);

    len_type m_min, m_max, n_min, n_max;
    std::tie(m_min, m_max, std::ignore,
             n_min, n_max, std::ignore) =
        comm.distribute_over_threads_2d(m, n, MB, NB);

    if (rs_min == 1 && (cs_min != 1 || cs_max < rs_max))
    {
        for (len_type j0 = n_min, jb = n_min/NB;j0 < n_max;j0 += NB, jb++)
        {
            len_type n_loc = std::min(NB, n_max-j0);

            for (len_type i0 = m_min, ib = m_min/MB;i0 < m_max;i0 += MB, ib++)
            {
                len_type m_loc = std::min(MB, m_max-i0);

                stride_type rs_B = rbs_B[ib];
                const T* p_B = B.data() + (rs_B ? rscat_B[i0] : 0);
                T* p_A = B.data() + rs_A*i0 + cs_A*j0;

                if (rs_B)
                {
                    TBLIS_SPECIAL_CASE(m_loc == MB,
                    TBLIS_SPECIAL_CASE(n_loc == NB,
                    TBLIS_SPECIAL_CASE(rs_A == 1,
                    TBLIS_SPECIAL_CASE(rs_B == 1,
                    TBLIS_SPECIAL_CASE(alpha == T(1),
                    TBLIS_SPECIAL_CASE(beta == T(0),
                    for (len_type j = 0;j < n_loc;j++)
                    {
                        for (len_type i = 0;i < m_loc;i++)
                        {
                            p_B[i*rs_B + cscat_B[j0+j]] =
                                alpha*p_A[i*rs_A + j*cs_A] +
                                beta*p_B[i*rs_B + cscat_B[j0+j]];
                        }
                    }
                    ))))));
                }
                else
                {
                    for (len_type j = 0;j < n_loc;j++)
                    {
                        for (len_type i = 0;i < m_loc;i++)
                        {
                            p_B[rscat_B[i0+i] + cscat_B[j0+j]] =
                                alpha*p_A[i*rs_A + j*cs_A] +
                                beta*p_B[rscat_B[i0+i] + cscat_B[j0+j]];
                        }
                    }
                }
            }
        }
    }
    else
    {
        for (len_type i0 = m_min, ib = m_min/MB;i0 < m_max;i0 += MB, ib++)
        {
            len_type m_loc = std::min(MB, m_max-i0);

            for (len_type j0 = n_min, jb = n_min/NB;j0 < n_max;j0 += NB, jb++)
            {
                len_type n_loc = std::min(NB, n_max-j0);

                stride_type cs_B = cbs_B[jb];
                const T* p_B = B.data() + (cs_B ? cscat_B[j0] : 0);
                T* p_A = B.data() + rs_A*i0 + cs_A*j0;

                if (cs_B)
                {
                    TBLIS_SPECIAL_CASE(m_loc == MB,
                    TBLIS_SPECIAL_CASE(n_loc == NB,
                    TBLIS_SPECIAL_CASE(cs_A == 1,
                    TBLIS_SPECIAL_CASE(cs_B == 1,
                    TBLIS_SPECIAL_CASE(alpha == T(1),
                    TBLIS_SPECIAL_CASE(beta == T(0),
                    for (len_type i = 0;i < m_loc;i++)
                    {
                        for (len_type j = 0;j < n_loc;j++)
                        {
                            p_B[rscat_B[i0+i] + j*cs_B] =
                                alpha*p_A[i*rs_A + j*cs_A] +
                                beta*p_B[rscat_B[i0+i] + j*cs_B];
                        }
                    }
                    ))))));
                }
                else
                {
                    for (len_type j = 0;j < n_loc;j++)
                    {
                        for (len_type i = 0;i < m_loc;i++)
                        {
                            p_B[rscat_B[i0+i] + cscat_B[j0+j]] =
                                alpha*p_A[i*rs_A + j*cs_A] +
                                beta*p_B[rscat_B[i0+i] + cscat_B[j0+j]];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void add(T alpha, matrix_view<const T> A, T beta, tensor_matrix<T> B)
{
    parallelize_if(
        [&](const communicator& comm)
        {
            add(comm, alpha, A, beta, B);
        },
        nullptr);
}

template <typename T>
void add(single_t, T alpha, matrix_view<const T> A, T beta, tensor_matrix<T> B)
{
    communicator comm;
    add(comm, alpha, A, beta, B);
}

}

#endif
