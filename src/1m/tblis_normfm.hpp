#ifndef _TBLIS_NORMFM_HPP_
#define _TBLIS_NORMFM_HPP_

#include "tblis.hpp"

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
T tblis_normfm(const_matrix_view<T> A);

template <typename T>
T tblis_normfm(const_scatter_matrix_view<T> A);

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
               stride_type rs_A, stride_type cs_A);

template <typename T>
T tblis_normfm(idx_type m, idx_type n,
               const T* A,
               const stride_type* rscat_A, stride_type cs_A);

template <typename T>
T tblis_normfm(idx_type m, idx_type n,
                  const T* A,
                  stride_type rs_A, const stride_type* cscat_A);

template <typename T>
T tblis_normfm(idx_type m, idx_type n,
               const T* A,
               const stride_type* rscat_A, const stride_type* cscat_A);

}

#endif
