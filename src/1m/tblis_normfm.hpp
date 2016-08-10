#ifndef _TBLIS_NORMFM_HPP_
#define _TBLIS_NORMFM_HPP_

#include "tblis_marray.hpp"
#include "tblis_thread.hpp"

namespace tblis
{

template <typename T>
void tblis_normfm_ref(thread_communicator& comm,
                      idx_type m, idx_type n,
                      const T* A,
                      stride_type rs_A, stride_type cs_A,
                      T& norm);

template <typename T>
void tblis_normfm_ref(thread_communicator& comm,
                      idx_type m, idx_type n,
                      const T* A,
                      const stride_type* rscat_A, stride_type cs_A,
                      T& norm);

template <typename T>
void tblis_normfm_ref(thread_communicator& comm,
                      idx_type m, idx_type n,
                      const T* A,
                      stride_type rs_A, const stride_type* cscat_A,
                      T& norm);

template <typename T>
void tblis_normfm_ref(thread_communicator& comm,
                      idx_type m, idx_type n,
                      const T* A,
                      const stride_type* rscat_A, const stride_type* cscat_A,
                      T& norm);

template <typename T>
void tblis_normfm(const_matrix_view<T> A, T& norm);

template <typename T>
void tblis_normfm(const_scatter_matrix_view<T> A, T& norm);

template <typename T>
T tblis_normfm(const_matrix_view<T> A)
{
    T norm;
    tblis_normfm(A, norm);
    return norm;
}

template <typename T>
T tblis_normfm(const_scatter_matrix_view<T> A)
{
    T norm;
    tblis_normfm(A, norm);
    return norm;
}

template <typename T>
void tblis_normfm(idx_type m, idx_type n,
                  const T* A,
                  stride_type rs_A, stride_type cs_A,
                  T& norm);

template <typename T>
void tblis_normfm(idx_type m, idx_type n,
                  const T* A,
                  const stride_type* rscat_A, stride_type cs_A,
                  T& norm);

template <typename T>
void tblis_normfm(idx_type m, idx_type n,
                  const T* A,
                  stride_type rs_A, const stride_type* cscat_A,
                  T& norm);

template <typename T>
void tblis_normfm(idx_type m, idx_type n,
                  const T* A,
                  const stride_type* rscat_A, const stride_type* cscat_A,
                  T& norm);

template <typename T>
T tblis_normfm(idx_type m, idx_type n,
               const T* A,
               stride_type rs_A, stride_type cs_A)
{
    T norm;
    tblis_normfm(m, n, A, rs_A, cs_A, norm);
    return norm;
}

template <typename T>
T tblis_normfm(idx_type m, idx_type n,
               const T* A,
               const stride_type* rscat_A, stride_type cs_A)
{
    T norm;
    tblis_normfm(m, n, A, rscat_A, cs_A, norm);
    return norm;
}

template <typename T>
T tblis_normfm(idx_type m, idx_type n,
                  const T* A,
                  stride_type rs_A, const stride_type* cscat_A)
{
    T norm;
    tblis_normfm(m, n, A, rs_A, cscat_A, norm);
    return norm;
}

template <typename T>
T tblis_normfm(idx_type m, idx_type n,
               const T* A,
               const stride_type* rscat_A, const stride_type* cscat_A)
{
    T norm;
    tblis_normfm(m, n, A, rscat_A, cscat_A, norm);
    return norm;
}

}

#endif
