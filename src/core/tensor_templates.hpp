#ifndef _TBLIS_TEMPLATES_HPP_
#define _TBLIS_TEMPLATES_HPP_

#include "tblis.hpp"

#include "util/util.hpp"

namespace tblis
{

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx,
                      typename C_ptr, typename C_len, typename C_stride, typename C_idx>
int tensor_mult(T alpha, A_ptr A, gint_t ndim_A, A_len len_A, A_stride stride_A, A_idx idx_A,
                         B_ptr B, gint_t ndim_B, B_len len_B, B_stride stride_B, B_idx idx_B,
                T  beta, C_ptr C, gint_t ndim_C, C_len len_C, C_stride stride_C, C_idx idx_C)
{
    Tensor<T> A_(ndim_A, len_A, A, stride_A);
    Tensor<T> B_(ndim_B, len_B, B, stride_B);
    Tensor<T> C_(ndim_C, len_C, C, stride_C);

    return tensor_mult(alpha, A_, util::ptr(idx_A),
                              B_, util::ptr(idx_B),
                        beta, C_, util::ptr(idx_C));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx,
                      typename C_ptr, typename C_len, typename C_stride, typename C_idx>
int tensor_contract(T alpha, A_ptr A, gint_t ndim_A, A_len len_A, A_stride stride_A, A_idx idx_A,
                             B_ptr B, gint_t ndim_B, B_len len_B, B_stride stride_B, B_idx idx_B,
                    T  beta, C_ptr C, gint_t ndim_C, C_len len_C, C_stride stride_C, C_idx idx_C)
{
    Tensor<T> A_(ndim_A, len_A, A, stride_A);
    Tensor<T> B_(ndim_B, len_B, B, stride_B);
    Tensor<T> C_(ndim_C, len_C, C, stride_C);

    return tensor_contract(alpha, A_, util::ptr(idx_A),
                                  B_, util::ptr(idx_B),
                            beta, C_, util::ptr(idx_C));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx,
                      typename C_ptr, typename C_len, typename C_stride, typename C_idx>
int tensor_weight(T alpha, A_ptr A, gint_t ndim_A, A_len len_A, A_stride stride_A, A_idx idx_A,
                           B_ptr B, gint_t ndim_B, B_len len_B, B_stride stride_B, B_idx idx_B,
                  T  beta, C_ptr C, gint_t ndim_C, C_len len_C, C_stride stride_C, C_idx idx_C)
{
    Tensor<T> A_(ndim_A, len_A, A, stride_A);
    Tensor<T> B_(ndim_B, len_B, B, stride_B);
    Tensor<T> C_(ndim_C, len_C, C, stride_C);

    return tensor_weight(alpha, A_, util::ptr(idx_A),
                                B_, util::ptr(idx_B),
                          beta, C_, util::ptr(idx_C));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx,
                      typename C_ptr, typename C_len, typename C_stride, typename C_idx>
int tensor_outer_prod(T alpha, A_ptr A, gint_t ndim_A, A_len len_A, A_stride stride_A, A_idx idx_A,
                               B_ptr B, gint_t ndim_B, B_len len_B, B_stride stride_B, B_idx idx_B,
                      T  beta, C_ptr C, gint_t ndim_C, C_len len_C, C_stride stride_C, C_idx idx_C)
{
    Tensor<T> A_(ndim_A, len_A, A, stride_A);
    Tensor<T> B_(ndim_B, len_B, B, stride_B);
    Tensor<T> C_(ndim_C, len_C, C, stride_C);

    return tensor_outer_prod(alpha, A_, util::ptr(idx_A),
                                    B_, util::ptr(idx_B),
                              beta, C_, util::ptr(idx_C));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx>
int tensor_sum(T alpha, A_ptr A, gint_t ndim_A, A_len len_A, A_stride stride_A, A_idx idx_A,
               T  beta, B_ptr B, gint_t ndim_B, B_len len_B, B_stride stride_B, B_idx idx_B)
{
    Tensor<T> A_(ndim_A, len_A, A, stride_A);
    Tensor<T> B_(ndim_B, len_B, B, stride_B);

    return tensor_sum(alpha, A_, util::ptr(idx_A),
                       beta, B_, util::ptr(idx_B));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx>
int tensor_trace(T alpha, A_ptr A, gint_t ndim_A, A_len len_A, A_stride stride_A, A_idx idx_A,
                 T  beta, B_ptr B, gint_t ndim_B, B_len len_B, B_stride stride_B, B_idx idx_B)
{
    Tensor<T> A_(ndim_A, len_A, A, stride_A);
    Tensor<T> B_(ndim_B, len_B, B, stride_B);

    return tensor_trace(alpha, A_, util::ptr(idx_A),
                         beta, B_, util::ptr(idx_B));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx>
int tensor_replicate(T alpha, A_ptr A, gint_t ndim_A, A_len len_A, A_stride stride_A, A_idx idx_A,
                     T  beta, B_ptr B, gint_t ndim_B, B_len len_B, B_stride stride_B, B_idx idx_B)
{
    Tensor<T> A_(ndim_A, len_A, A, stride_A);
    Tensor<T> B_(ndim_B, len_B, B, stride_B);

    return tensor_replicate(alpha, A_, util::ptr(idx_A),
                             beta, B_, util::ptr(idx_B));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx>
int tensor_transpose(T alpha, A_ptr A, gint_t ndim_A, A_len len_A, A_stride stride_A, A_idx idx_A,
                     T  beta, B_ptr B, gint_t ndim_B, B_len len_B, B_stride stride_B, B_idx idx_B)
{
    Tensor<T> A_(ndim_A, len_A, A, stride_A);
    Tensor<T> B_(ndim_B, len_B, B, stride_B);

    return tensor_transpose(alpha, A_, util::ptr(idx_A),
                             beta, B_, util::ptr(idx_B));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx>
T tensor_dot(A_ptr A, gint_t ndim_A, A_len len_A, A_stride stride_A, A_idx idx_A,
             B_ptr B, gint_t ndim_B, B_len len_B, B_stride stride_B, B_idx idx_B)
{
    Tensor<T> A_(ndim_A, len_A, A, stride_A);
    Tensor<T> B_(ndim_B, len_B, B, stride_B);

    return tensor_dot(A_, util::ptr(idx_A),
                      B_, util::ptr(idx_B));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx>
int tensor_dot(A_ptr A, gint_t ndim_A, A_len len_A, A_stride stride_A, A_idx idx_A,
               B_ptr B, gint_t ndim_B, B_len len_B, B_stride stride_B, B_idx idx_B, T& val)
{
    Tensor<T> A_(ndim_A, len_A, A, stride_A);
    Tensor<T> B_(ndim_B, len_B, B, stride_B);

    return tensor_dot(A_, util::ptr(idx_A),
                      B_, util::ptr(idx_B), val);
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx>
int tensor_scale(T alpha, A_ptr A, gint_t ndim_A, A_len len_A, A_stride stride_A, A_idx idx_A)
{
    Tensor<T> A_(ndim_A, len_A, A, stride_A);
    return tensor_scale(alpha, A_, util::ptr(idx_A));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx>
std::pair<T,inc_t> tensor_reduce(reduce_t op, A_ptr A, gint_t ndim_A, A_len len_A, A_stride stride_A, A_idx idx_A)
{
    Tensor<T> A_(ndim_A, len_A, A, stride_A);
    return tensor_reduce(op, A_, util::ptr(idx_A));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx>
T tensor_reduce(reduce_t op, A_ptr A, gint_t ndim_A, A_len len_A, A_stride stride_A, A_idx idx_A, inc_t& idx)
{
    Tensor<T> A_(ndim_A, len_A, A, stride_A);
    return tensor_reduce(op, A_, util::ptr(idx_A), idx);
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx>
int tensor_reduce(reduce_t op, A_ptr A, gint_t ndim_A, A_len len_A, A_stride stride_A, A_idx idx_A, T& val)
{
    Tensor<T> A_(ndim_A, len_A, A, stride_A);
    return tensor_reduce(op, A_, util::ptr(idx_A), val);
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx>
int tensor_reduce(reduce_t op, A_ptr A, gint_t ndim_A, A_len len_A, A_stride stride_A, A_idx idx_A, T& val, inc_t& idx)
{
    Tensor<T> A_(ndim_A, len_A, A, stride_A);
    return tensor_reduce(op, A_, util::ptr(idx_A), val, idx);
}

}

#endif
