#include "tblis_matrix_reduce.hpp"

namespace tblis
{

template <typename T>
void tblis_normfm_ref(thread_communicator& comm,
                      len_type m, len_type n,
                      const T* TBLIS_RESTRICT A,
                      stride_type rs_A, stride_type cs_A,
                      T& TBLIS_RESTRICT norm)
{
    T subnrm = T();

    len_type m_min, m_max;
    std::tie(m_min, m_max, std::ignore) = comm.distribute_over_threads(m);

    for (len_type i = m_min;i < m_max;i++)
    {
        for (len_type j = 0;j < n;j++)
        {
            T val = *(A+rs_A*i+cs_A*j);
            subnrm += norm2(val);
        }
    }

    comm.reduce(subnrm);
    norm = sqrt(real(subnrm));
}

template <typename T>
void tblis_normfm_ref(thread_communicator& comm,
                      len_type m, len_type n,
                      const T* TBLIS_RESTRICT A,
                      const stride_type* TBLIS_RESTRICT rscat_A, stride_type cs_A,
                      T& TBLIS_RESTRICT norm)
{
    T subnrm = T();

    len_type m_min, m_max;
    std::tie(m_min, m_max, std::ignore) = comm.distribute_over_threads(m);

    for (len_type i = m_min;i < m_max;i++)
    {
        for (len_type j = 0;j < n;j++)
        {
            T val = *(A+rscat_A[i]+cs_A*j);
            subnrm += norm2(val);
        }
    }

    comm.reduce(subnrm);
    norm = sqrt(real(subnrm));
}

template <typename T>
void tblis_normfm_ref(thread_communicator& comm,
                      len_type m, len_type n,
                      const T* TBLIS_RESTRICT A,
                      stride_type rs_A, const stride_type* TBLIS_RESTRICT cscat_A,
                      T& TBLIS_RESTRICT norm)
{
    T subnrm = T();

    len_type m_min, m_max;
    std::tie(m_min, m_max, std::ignore) = comm.distribute_over_threads(m);

    for (len_type i = m_min;i < m_max;i++)
    {
        for (len_type j = 0;j < n;j++)
        {
            T val = *(A+rs_A*i+cscat_A[j]);
            subnrm += norm2(val);
        }
    }

    comm.reduce(subnrm);
    norm = sqrt(real(subnrm));
}

template <typename T>
void tblis_normfm_ref(thread_communicator& comm,
                      len_type m, len_type n,
                      const T* TBLIS_RESTRICT A,
                      const stride_type* TBLIS_RESTRICT rscat_A,
                      const stride_type* TBLIS_RESTRICT cscat_A,
                      T& TBLIS_RESTRICT norm)
{
    T subnrm = T();

    len_type m_min, m_max;
    std::tie(m_min, m_max, std::ignore) = comm.distribute_over_threads(m);

    for (len_type i = m_min;i < m_max;i++)
    {
        for (len_type j = 0;j < n;j++)
        {
            T val = *(A+rscat_A[i]+cscat_A[j]);
            subnrm += norm2(val);
        }
    }

    comm.reduce(subnrm);
    norm = sqrt(real(subnrm));
}

template <typename T>
void tblis_normfm(const_matrix_view<T> A, T& norm)
{
    tblis_normfm(A.length(0), A.length(1),
                 A.data(),
                 A.stride(0), A.stride(1),
                 norm);
}

template <typename T>
void tblis_normfm(const_scatter_matrix_view<T> A, T& norm)
{
    stride_type rs = A.stride(0);
    stride_type cs = A.stride(1);
    const stride_type* rscat = A.scatter(0);
    const stride_type* cscat = A.scatter(1);

    if (rs == 0 && cs == 0)
    {
        tblis_normfm(A.length(0), A.length(1),
                     A.data(),
                     rscat, cscat,
                     norm);
    }
    else if (rs == 0)
    {
        tblis_normfm(A.length(0), A.length(1),
                     A.data(),
                     rscat, cs,
                     norm);
    }
    else if (cs == 0)
    {
        tblis_normfm(A.length(0), A.length(1),
                     A.data(),
                     rs, cscat,
                     norm);
    }
    else
    {
        tblis_normfm(A.length(0), A.length(1),
                     A.data(),
                     rs, cs,
                     norm);
    }
}

template <typename T>
void tblis_normfm(len_type m, len_type n,
                  const T* A,
                  stride_type rs_A, stride_type cs_A,
                  T& norm)
{
    parallelize
    (
        [&](thread_communicator& comm)
        {
            tblis_normfm_ref(comm, m, n, A, rs_A, cs_A, norm);
        }
    );
}

template <typename T>
void tblis_normfm(len_type m, len_type n,
                  const T* A,
                  const stride_type* rscat_A, stride_type cs_A,
                  T& norm)
{
    parallelize
    (
        [&](thread_communicator& comm)
        {
            tblis_normfm_ref(comm, m, n, A, rscat_A, cs_A, norm);
        }
    );
}

template <typename T>
void tblis_normfm(len_type m, len_type n,
                  const T* A,
                  stride_type rs_A, const stride_type* cscat_A,
                  T& norm)
{
    parallelize
    (
        [&](thread_communicator& comm)
        {
            tblis_normfm_ref(comm, m, n, A, rs_A, cscat_A, norm);
        }
    );
}

template <typename T>
void tblis_normfm(len_type m, len_type n,
                  const T* A,
                  const stride_type* rscat_A, const stride_type* cscat_A,
                  T& norm)
{
    parallelize
    (
        [&](thread_communicator& comm)
        {
            tblis_normfm_ref(comm, m, n, A, rscat_A, cscat_A, norm);
        }
    );
}

#define INSTANTIATE_FOR_TYPE(T) \
template void tblis_normfm_ref(thread_communicator& comm, idx_type m, idx_type n, const T* A,       stride_type     rs_A,       stride_type     cs_A, T& norm); \
template void tblis_normfm_ref(thread_communicator& comm, idx_type m, idx_type n, const T* A, const stride_type* rscat_A,       stride_type     cs_A, T& norm); \
template void tblis_normfm_ref(thread_communicator& comm, idx_type m, idx_type n, const T* A,       stride_type     rs_A, const stride_type* cscat_A, T& norm); \
template void tblis_normfm_ref(thread_communicator& comm, idx_type m, idx_type n, const T* A, const stride_type* rscat_A, const stride_type* cscat_A, T& norm); \
template void tblis_normfm(        const_matrix_view<T> A, T& norm); \
template void tblis_normfm(const_scatter_matrix_view<T> A, T& norm); \
template void tblis_normfm(idx_type m, idx_type n, const T* A,       stride_type     rs_A,       stride_type     cs_A, T& norm); \
template void tblis_normfm(idx_type m, idx_type n, const T* A, const stride_type* rscat_A,       stride_type     cs_A, T& norm); \
template void tblis_normfm(idx_type m, idx_type n, const T* A,       stride_type     rs_A, const stride_type* cscat_A, T& norm); \
template void tblis_normfm(idx_type m, idx_type n, const T* A, const stride_type* rscat_A, const stride_type* cscat_A, T& norm);
#include "tblis_instantiate_for_types.hpp"

}
