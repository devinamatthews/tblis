#ifndef _TBLIS_TENSOR_IFACE_HPP_
#define _TBLIS_TENSOR_IFACE_HPP_

#include "tblis.hpp"

#include "impl/tensor_impl.hpp"

#include <vector>
#include <string>
#include <utility>

namespace tblis
{

template <typename T> class Tensor;

/**
 * Multiply two tensors together and sum onto a third
 *
 * This form generalizes contraction and weighting with the unary operations trace, transpose, and replicate. Note that
 * the binary contraction operation is similar in form to the unary trace operation, while the binary weighting operation is similar in form to the
 * unary diagonal operation. Any combination of these operations may be performed.
 */
template <typename T>
int tensor_mult(T alpha, const Tensor<T>& A, std::string idx_A,
                         const Tensor<T>& B, std::string idx_B,
                T  beta,       Tensor<T>& C, std::string idx_C)
{
    #if TENSOR_ERROR_CHECKING
    util::check_tensor_indices(A, idx_A, B, idx_B, C, idx_C,
                               true, true, true,
                               true, true, true,
                               true);
    #endif

    std::string idx_A_, idx_B_, idx_C_;
    Tensor<T> A_, B_, C_;

    LockedDiagonal(A, idx_A, A_, idx_A_);
    LockedDiagonal(B, idx_B, B_, idx_B_);
          Diagonal(C, idx_C, C_, idx_C_);

    return impl::tensor_mult_impl(alpha, A_, idx_A_,
                                         B_, idx_B_,
                                   beta, C_, idx_C_);
}

/**
 * Contract two tensors into a third
 *
 * The general form for a contraction is ab...ef... * ef...cd... -> ab...cd... where the indices ef... will be summed over.
 * Indices may be transposed in any tensor. Any index group may be empty (in the case that ef... is empty, this reduces to an outer product).
 */
template <typename T>
int tensor_contract(T alpha, const Tensor<T>& A, std::string idx_A,
                             const Tensor<T>& B, std::string idx_B,
                    T  beta,       Tensor<T>& C, std::string idx_C)
{
    #if TENSOR_ERROR_CHECKING
    util::check_tensor_indices(A, idx_A, B, idx_B, C, idx_C,
                               false, false, false,
                               true, true, true,
                               false);
    #endif

    std::string idx_A_, idx_B_, idx_C_;
    Tensor<T> A_, B_, C_;

    LockedDiagonal(A, idx_A, A_, idx_A_);
    LockedDiagonal(B, idx_B, B_, idx_B_);
          Diagonal(C, idx_C, C_, idx_C_);

    return impl::tensor_contract_impl(alpha, A_, idx_A_,
                                             B_, idx_B_,
                                       beta, C_, idx_C_);
}

/**
 * Weight a tensor by a second and sum onto a third
 *
 * The general form for a weighting is ab...ef... * ef...cd... -> ab...cd...ef... with no indices being summed over.
 * Indices may be transposed in any tensor. Any index group may be empty
 * (in the case that ef... is empty, this reduces to an outer product).
 */
template <typename T>
int tensor_weight(T alpha, const Tensor<T>& A, std::string idx_A,
                           const Tensor<T>& B, std::string idx_B,
                  T  beta,       Tensor<T>& C, std::string idx_C)
{
    #if TENSOR_ERROR_CHECKING
    util::check_tensor_indices(A, idx_A, B, idx_B, C, idx_C,
                               false, false, false,
                               false, true, true,
                               true);
    #endif

    std::string idx_A_, idx_B_, idx_C_;
    Tensor<T> A_, B_, C_;

    LockedDiagonal(A, idx_A, A_, idx_A_);
    LockedDiagonal(B, idx_B, B_, idx_B_);
          Diagonal(C, idx_C, C_, idx_C_);

    return impl::tensor_weight_impl(alpha, A_, idx_A_,
                                           B_, idx_B_,
                                     beta, C_, idx_C_);
}

/**
 * Sum the outer product of two tensors onto a third
 *
 * The general form for an outer product is ab... * cd... -> ab...cd... with no indices being summed over.
 * Indices may be transposed in any tensor.
 */
template <typename T>
int tensor_outer_prod(T alpha, const Tensor<T>& A, std::string idx_A,
                               const Tensor<T>& B, std::string idx_B,
                      T  beta,       Tensor<T>& C, std::string idx_C)
{
    #if TENSOR_ERROR_CHECKING
    util::check_tensor_indices(A, idx_A, B, idx_B, C, idx_C,
                               false, false, false,
                               false, true, true,
                               false);
    #endif

    std::string idx_A_, idx_B_, idx_C_;
    Tensor<T> A_, B_, C_;

    LockedDiagonal(A, idx_A, A_, idx_A_);
    LockedDiagonal(B, idx_B, B_, idx_B_);
          Diagonal(C, idx_C, C_, idx_C_);

    return impl::tensor_outer_prod_impl(alpha, A_, idx_A_,
                                               B_, idx_B_,
                                         beta, C_, idx_C_);
}

/**
 * sum a tensor (presumably operated on in one or more ways) onto a second
 *
 * This form generalizes all of the unary operations trace, transpose, and replicate, which may be performed
 * in any combination.
 */
template <typename T>
int tensor_sum(T alpha, const Tensor<T>& A, std::string idx_A,
               T  beta,       Tensor<T>& B, std::string idx_B)
{
    #if TENSOR_ERROR_CHECKING
    util::check_tensor_indices(A, idx_A, B, idx_B,
                               true, true, true);
    #endif

    std::string idx_A_, idx_B_;
    Tensor<T> A_, B_;

    LockedDiagonal(A, idx_A, A_, idx_A_);
          Diagonal(B, idx_B, B_, idx_B_);

    return impl::tensor_sum_impl(alpha, A_, idx_A_,
                                  beta, B_, idx_B_);
}

/**
 * Sum over (semi)diagonal elements of a tensor and sum onto a second
 *
 * The general form for a trace operation is ab...k*l*... -> ab... where k* denotes the index k appearing one or more times, etc. and where
 * the indices kl... will be summed (traced) over. Indices may be transposed, and multiple appearances
 * of the traced indices kl... need not appear together. Either set of indices may be empty, with the special case that when no indices
 * are traced over, the result is the same as transpose.
 */
template <typename T>
int tensor_trace(T alpha, const Tensor<T>& A, std::string idx_A,
                 T  beta,       Tensor<T>& B, std::string idx_B)
{
    #if TENSOR_ERROR_CHECKING
    util::check_tensor_indices(A, idx_A, B, idx_B,
                               true, false, true);
    #endif

    std::string idx_A_, idx_B_;
    Tensor<T> A_, B_;

    LockedDiagonal(A, idx_A, A_, idx_A_);
          Diagonal(B, idx_B, B_, idx_B_);

    return impl::tensor_trace_impl(alpha, A_, idx_A_,
                                    beta, B_, idx_B_);
}

/**
 * Replicate a tensor and sum onto a second
 *
 * The general form for a replication operation is ab... -> ab...c*d*... where c* denotes the index c appearing one or more times.
 * Any indices may be transposed.
 */
template <typename T>
int tensor_replicate(T alpha, const Tensor<T>& A, std::string idx_A,
                     T  beta,       Tensor<T>& B, std::string idx_B)
{
    #if TENSOR_ERROR_CHECKING
    util::check_tensor_indices(A, idx_A, B, idx_B,
                               false, true, true);
    #endif

    std::string idx_A_, idx_B_;
    Tensor<T> A_, B_;

    LockedDiagonal(A, idx_A, A_, idx_A_);
          Diagonal(B, idx_B, B_, idx_B_);

    return impl::tensor_replicate_impl(alpha, A_, idx_A_,
                                        beta, B_, idx_B_);
}

/**
 * Transpose a tensor and sum onto a second
 *
 * The general form for a transposition operation is ab... -> P(ab...) where P is some permutation. Transposition may change
 * the order in which the elements of the tensor are physically stored.
 */
template <typename T>
int tensor_transpose(T alpha, const Tensor<T>& A, std::string idx_A,
                     T  beta,       Tensor<T>& B, std::string idx_B)
{
    #if TENSOR_ERROR_CHECKING
    util::check_tensor_indices(A, idx_A, B, idx_B,
                               false, false, true);
    #endif

    std::string idx_A_, idx_B_;
    Tensor<T> A_, B_;

    LockedDiagonal(A, idx_A, A_, idx_A_);
          Diagonal(B, idx_B, B_, idx_B_);

    return impl::tensor_transpose_impl(alpha, A_, idx_A_,
                                        beta, B_, idx_B_);
}

/**
 * Return the dot product of two tensors
 */
template <typename T>
T tensor_dot(const Tensor<T>& A, std::string idx_A,
             const Tensor<T>& B, std::string idx_B)
{
    T val;
    int ret = tensor_dot(A, idx_A, B, idx_B, val);
    return val;
}

template <typename T>
int tensor_dot(const Tensor<T>& A, std::string idx_A,
               const Tensor<T>& B, std::string idx_B, T& val)
{
    #if TENSOR_ERROR_CHECKING
    util::check_tensor_indices(A, idx_A, B, idx_B,
                               false, false, true);
    #endif

    std::string idx_A_, idx_B_;
    Tensor<T> A_, B_;

    LockedDiagonal(A, idx_A, A_, idx_A_);
    LockedDiagonal(B, idx_B, B_, idx_B_);

    return impl::tensor_dot_impl(A_, idx_A_,
                                 B_, idx_B_, val);
}

/**
 * Scale a tensor by a scalar
 */
template <typename T>
int tensor_scale(T alpha, Tensor<T>& A, std::string idx_A)
{
    #if TENSOR_ERROR_CHECKING
    util::check_tensor_indices(A, idx_A);
    #endif

    std::string idx_A_;
    Tensor<T> A_;

    Diagonal(A, idx_A, A_, idx_A_);

    return impl::tensor_scale_impl(alpha, A_, idx_A_);
}

/**
 * Return the reduction of a tensor, along with the corresponding index (as an offset from A) for MAX, MIN, MAX_ABS, and MIN_ABS reductions
 */
template <typename T>
std::pair<T,inc_t> tensor_reduce(reduce_t op, const Tensor<T>& A, std::string idx_A)
{
    std::pair<T,inc_t> p;
    int ret = tensor_reduce(op, A, idx_A, p.first, p.second);
    return p;
}

template <typename T>
T tensor_reduce(reduce_t op, const Tensor<T>& A, std::string idx_A, inc_t& idx)
{
    T val;
    int ret = tensor_reduce(op, A, idx_A, val, idx);
    return val;
}

template <typename T>
int tensor_reduce(reduce_t op, const Tensor<T>& A, std::string idx_A, T& val)
{
   inc_t idx;
   return tensor_reduce(op, A, idx_A, val, idx);
}

template <typename T>
int tensor_reduce(reduce_t op, const Tensor<T>& A, std::string idx_A, T& val, inc_t& idx)
{
    #if TENSOR_ERROR_CHECKING
    util::check_tensor_indices(A, idx_A);
    #endif

    std::string idx_A_;
    Tensor<T> A_;

    LockedDiagonal(A, idx_A, A_, idx_A_);

    return impl::tensor_reduce_impl(op, A_, idx_A_, val, idx);
}

template <typename len_type, typename stride_type>
siz_t tensor_size(gint_t ndim, const len_type len, const stride_type stride)
{
    #if TENSOR_ERROR_CHECKING
    util::check_tensor(ndim, len, stride);
    #endif

    siz_t size = 1;

    for (gint_t i = 0;i < ndim;i++)
    {
        size *= len[i];
    }

    return size;
}

template <typename len_type>
siz_t tensor_size(gint_t ndim, const len_type len)
{
    return tensor_size(ndim, len, (inc_t*)NULL);
}

template <typename len_type, typename stride_type>
siz_t tensor_storage_size(gint_t ndim, const len_type len, const stride_type stride)
{
    if (!stride)
    {
       return tensor_size(ndim, len, stride);
    }

    #if TENSOR_ERROR_CHECKING
    util::check_tensor(ndim, len, stride);
    #endif

    inc_t min_idx = 0;
    inc_t max_idx = 0;

    for (gint_t i = 0;i < ndim;i++)
    {
        if (stride[i] < 0)
        {
            min_idx += stride[i]*(len[i]-1);
        }
        else
        {
            max_idx += stride[i]*(len[i]-1);
        }
    }

    return (siz_t)(max_idx - min_idx + 1);
}

}

#endif
