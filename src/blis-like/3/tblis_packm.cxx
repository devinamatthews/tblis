#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

namespace detail
{
    MemoryPool BuffersForA(4096);
    MemoryPool BuffersForB(4096);
}

template <typename T, dim_t MR, dim_t KR>
void PackMicroPanel(dim_t m, dim_t k,
                    const T* restrict & p_a, inc_t rs_a, inc_t cs_a,
                    T* restrict & p_ap)
{
    dim_t k_rem = util::remainder(k, KR);

    if (m == MR)
    {
        if (rs_a == 1)
        {
            for (dim_t i = 0;i < k;i++)
            {
                for (dim_t mr = 0;mr < MR;mr++)
                {
                    p_ap[mr] = p_a[mr];
                }

                p_a += cs_a;
                p_ap += MR;
            }
        }
        else if (cs_a == 1)
        {
            for (dim_t mr = 0;mr < MR;mr++)
            {
                for (dim_t i = 0;i < k;i++)
                {
                    p_ap[MR*i+mr] = p_a[i+rs_a*mr];
                }
            }

            p_a += cs_a*k;
            p_ap += MR*k;
        }
        else
        {
            for (dim_t i = 0;i < k;i++)
            {
                for (dim_t mr = 0;mr < MR;mr++)
                {
                    p_ap[mr] = p_a[rs_a*mr];
                }

                p_a += cs_a;
                p_ap += MR;
            }
        }
    }
    else
    {
        for (dim_t i = 0;i < k;i++)
        {
            for (dim_t mr = 0;mr < m;mr++)
            {
                p_ap[mr] = p_a[rs_a*mr];
            }

            for (dim_t mr = m;mr < MR;mr++)
            {
                p_ap[mr] = T();
            }

            p_a += cs_a;
            p_ap += MR;
        }
    }

    for (dim_t i = 0;i < k_rem;i++)
    {
        for (dim_t mr = 0;mr < MR;mr++)
        {
            p_ap[mr] = T();
        }

        p_ap += MR;
    }

    p_a += rs_a*m - cs_a*k;
}

template <typename T, dim_t MR, dim_t KR>
void PackMicroPanel(dim_t m, dim_t k,
                    const T* restrict & p_a,
                    const inc_t* restrict & rs_a, inc_t cs_a,
                    T* restrict & p_ap)
{
    dim_t k_rem = util::remainder(k, KR);

    if (m == MR)
    {
        for (dim_t i = 0;i < k;i++)
        {
            for (dim_t mr = 0;mr < MR;mr++)
            {
                p_ap[mr] = p_a[rs_a[mr]];
            }

            p_a += cs_a;
            p_ap += MR;
        }
    }
    else
    {
        for (dim_t i = 0;i < k;i++)
        {
            for (dim_t mr = 0;mr < m;mr++)
            {
                p_ap[mr] = p_a[rs_a[mr]];
            }

            for (dim_t mr = m;mr < MR;mr++)
            {
                p_ap[mr] = T();
            }

            p_a += cs_a;
            p_ap += MR;
        }
    }

    for (dim_t i = 0;i < k_rem;i++)
    {
        for (dim_t mr = 0;mr < MR;mr++)
        {
            p_ap[mr] = T();
        }

        p_ap += MR;
    }

    p_a -= cs_a*k;
    rs_a += m;
}

template <typename T, dim_t MR, dim_t KR>
void PackMicroPanel(dim_t m, dim_t k,
                    const T* restrict & p_a,
                    inc_t rs_a, const inc_t* restrict cs_a,
                    T* restrict & p_ap)
{
    dim_t k_rem = util::remainder(k, KR);

    if (m == MR)
    {
        for (dim_t i = 0;i < k;i++)
        {
            for (dim_t mr = 0;mr < MR;mr++)
            {
                p_ap[mr] = p_a[cs_a[i] + rs_a*mr];
            }

            p_ap += MR;
        }
    }
    else
    {
        for (dim_t i = 0;i < k;i++)
        {
            for (dim_t mr = 0;mr < m;mr++)
            {
                p_ap[mr] = p_a[cs_a[i] + rs_a*mr];
            }

            for (dim_t mr = m;mr < MR;mr++)
            {
                p_ap[mr] = T();
            }

            p_ap += MR;
        }
    }

    for (dim_t i = 0;i < k_rem;i++)
    {
        for (dim_t mr = 0;mr < MR;mr++)
        {
            p_ap[mr] = T();
        }

        p_ap += MR;
    }

    p_a += rs_a*m;
}

template <typename T, dim_t MR, dim_t KR>
void PackMicroPanel(dim_t m, dim_t k,
                    const T* restrict & p_a,
                    const inc_t* restrict & rs_a, const inc_t* restrict cs_a,
                    T* restrict & p_ap)
{
    dim_t k_rem = util::remainder(k, KR);

    if (m == MR)
    {
        for (dim_t i = 0;i < k;i++)
        {
            for (dim_t mr = 0;mr < MR;mr++)
            {
                p_ap[mr] = p_a[cs_a[i] + rs_a[mr]];
            }

            p_ap += MR;
        }
    }
    else
    {
        for (dim_t i = 0;i < k;i++)
        {
            for (dim_t mr = 0;mr < m;mr++)
            {
                p_ap[mr] = p_a[cs_a[i] + rs_a[mr]];
            }

            for (dim_t mr = m;mr < MR;mr++)
            {
                p_ap[mr] = T();
            }

            p_ap += MR;
        }
    }

    for (dim_t i = 0;i < k_rem;i++)
    {
        for (dim_t mr = 0;mr < MR;mr++)
        {
            p_ap[mr] = T();
        }

        p_ap += MR;
    }

    rs_a += m;
}

template <typename T, dim_t MR, dim_t KR, bool Trans>
void PackRowPanel<T,MR,KR,Trans>::operator()(ThreadCommunicator& comm, const Matrix<T>& A, Matrix<T>& Ap) const
{
    dim_t m_a = (Trans ? A.width () : A.length());
    dim_t k_a = (Trans ? A.length() : A.width ());
    inc_t rs_a = (Trans ? A.col_stride() : A.row_stride());
    inc_t cs_a = (Trans ? A.row_stride() : A.col_stride());
    const T* p_a = A.data();
    T* p_ap = Ap.data();

    dim_t off_first, off_last;
    std::tie(off_first, off_last) = comm.distribute_over_threads(m_a, MR);

    //printf_locked("%s: (%d,%d) %ld %ld %ld\n", (Trans ? "B" : "A"), comm.gang_num(), comm.thread_num(), m_a, off_first, off_last);

    p_a += off_first*rs_a;
    p_ap += off_first*util::round_up(k_a, KR);

    for (dim_t off_m = off_first;off_m < off_last;off_m += MR)
    {
        PackMicroPanel<T,MR,KR>(std::min(MR, off_last-off_m), k_a,
                                p_a, rs_a, cs_a, p_ap);
    }

    /*
    comm.barrier();

    if (comm.thread_num() == 0)
    {
        printf("%s: %.15f\n", (Trans ? "B" : "A"), tblis_normfv(Ap.length()*Ap.width(), Ap.data(), 1));
    }
    */
}

template <typename T, dim_t MR, dim_t KR, bool Trans>
void PackRowPanel<T,MR,KR,Trans>::operator()(ThreadCommunicator& comm, const const_scatter_matrix_view<T>& A, Matrix<T>& Ap) const
{
    dim_t m_a = (Trans ? A.width () : A.length());
    dim_t k_a = (Trans ? A.length() : A.width ());
    inc_t rs_a = (Trans ? A.col_stride() : A.row_stride());
    inc_t cs_a = (Trans ? A.row_stride() : A.col_stride());
    const inc_t* rscat_a = (Trans ? A.col_scatter() : A.row_scatter());
    const inc_t* cscat_a = (Trans ? A.row_scatter() : A.col_scatter());
    const T* p_a = A.data();
    T* p_ap = Ap.data();

    dim_t off_first, off_last;
    std::tie(off_first, off_last) = comm.distribute_over_threads(m_a, MR);

    //printf_locked("%s: %d %ld %ld %ld\n", (Trans ? "B" : "A"), comm.thread_num(), m_a, off_first, off_last);

    p_a += off_first*rs_a;
    rscat_a += off_first;
    p_ap += off_first*util::round_up(k_a, KR);

    for (dim_t off_m = off_first;off_m < off_last;off_m += MR)
    {
        if (rs_a == 0 && cs_a == 0)
        {
            PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                    p_a, rscat_a, cscat_a, p_ap);
        }
        else if (rs_a == 0)
        {
            PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                    p_a, rscat_a, cs_a, p_ap);
        }
        else if (cs_a == 0)
        {
            PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                    p_a, rs_a, cscat_a, p_ap);
        }
        else
        {
            PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                    p_a, rs_a, cs_a, p_ap);
        }
    }

    //comm.barrier();

    //if (comm.thread_num() == 0)
    //{
    //    printf_locked("%s: %.15f\n", (Trans ? "B" : "A"), tblis_normfv(Ap.length()*Ap.width(), Ap.data(), 1));
    //}
}

static Mutex pack_lock;

template <typename T, dim_t MR, dim_t KR, bool Trans>
void PackRowPanel<T,MR,KR,Trans>::operator()(ThreadCommunicator& comm, const block_scatter_matrix<T,(Trans ? KR : MR),(Trans ? MR : KR)>& A_, Matrix<T>& Ap) const
{
    block_scatter_matrix<T,(Trans ? KR : MR),(Trans ? MR : KR)> A(A_);

    dim_t m_a = (Trans ? A.width () : A.length());
    dim_t k_a = (Trans ? A.length() : A.width ());
    T* p_ap = Ap.data();

    //{
    //std::lock_guard<Mutex> guard(pack_lock);

    dim_t off_first, off_last;
    std::tie(off_first, off_last) = comm.distribute_over_threads(m_a, MR);

    //printf_locked("%s: (%d,%d) %ld %ld %ld\n", (Trans ? "B" : "A"), comm.gang_num(), comm.thread_num(), m_a, off_first, off_last);

    p_ap += off_first*util::round_up(k_a, KR);

    (Trans ? A.width(MR) : A.length(MR));
    (Trans ? A.shift_right(off_first) : A.shift_down(off_first));

    //printf_locked("%s: (%d,%d) %p %p\n", (Trans ? "B" : "A"), comm.gang_num(), comm.thread_num(), A.data(), p_ap);

    for (dim_t off_m = off_first;off_m < off_last;off_m += MR)
    {
        inc_t rs_a = (Trans ? A.col_stride() : A.row_stride());
        inc_t cs_a = (Trans ? A.row_stride() : A.col_stride());
        const inc_t* rscat_a = (Trans ? A.col_scatter() : A.row_scatter());
        const inc_t* cscat_a = (Trans ? A.row_scatter() : A.col_scatter());
        const T* p_a = A.data();

        //printf("%d: %d %ld %ld %p %p\n", comm.thread_num(), off_m, rs_a, cs_a, rscat_a, cscat_a);

        if (rs_a == 0 && cs_a == 0)
        {
            PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                    p_a, rscat_a, cscat_a, p_ap);
        }
        else if (rs_a == 0)
        {
            PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                    p_a, rscat_a, cs_a, p_ap);
        }
        else if (cs_a == 0)
        {
            PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                    p_a, rs_a, cscat_a, p_ap);
        }
        else
        {
            PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                    p_a, rs_a, cs_a, p_ap);
        }

        (Trans ? A.shift_right() : A.shift_down());
    }

    (Trans ? A.shift_left(off_last) : A.shift_up(off_last));
    (Trans ? A.width(m_a) : A.length(m_a));

    //}

    //comm.barrier();

    //if (comm.thread_num() == 0)
    //{
    //    printf_locked("%s: (%d,%d) %.15f\n", (Trans ? "B" : "A"), comm.gang_num(), comm.thread_num(), tblis_normfv(Ap.length()*Ap.width(), Ap.data(), 1));
    //}
}

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void PackMicroPanel<T,          MR, KR>(dim_t m, dim_t k, const T* restrict & p_a, inc_t rs_a, inc_t cs_a, T* restrict & p_ap); \
template void PackMicroPanel<T,(MR==NR?0:NR),KR>(dim_t m, dim_t k, const T* restrict & p_a, inc_t rs_a, inc_t cs_a, T* restrict & p_ap);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void PackMicroPanel<T,          MR, KR>(dim_t m, dim_t k, const T* restrict & p_a, const inc_t* restrict & rs_a, inc_t cs_a, T* restrict & p_ap); \
template void PackMicroPanel<T,(MR==NR?0:NR),KR>(dim_t m, dim_t k, const T* restrict & p_a, const inc_t* restrict & rs_a, inc_t cs_a, T* restrict & p_ap);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void PackMicroPanel<T,          MR, KR>(dim_t m, dim_t k, const T* restrict & p_a, inc_t rs_a, const inc_t* restrict  cs_a, T* restrict & p_ap); \
template void PackMicroPanel<T,(MR==NR?0:NR),KR>(dim_t m, dim_t k, const T* restrict & p_a, inc_t rs_a, const inc_t* restrict  cs_a, T* restrict & p_ap);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void PackMicroPanel<T,          MR, KR>(dim_t m, dim_t k, const T* restrict & p_a, const inc_t* restrict & rs_a, const inc_t* restrict cs_a, T* restrict & p_ap); \
template void PackMicroPanel<T,(MR==NR?0:NR),KR>(dim_t m, dim_t k, const T* restrict & p_a, const inc_t* restrict & rs_a, const inc_t* restrict cs_a, T* restrict & p_ap);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template struct PackRowPanel<T,MR,KR,false>; \
template struct PackRowPanel<T,NR,KR, true>;
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

}
}
