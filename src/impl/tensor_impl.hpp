#ifndef _TBLIS_IMPL_HPP_
#define _TBLIS_IMPL_HPP_

#include "tblis.hpp"

#include "impl/tensor_impl_reference.hpp"
#include "impl/tensor_impl_blas.hpp"
#include "impl/tensor_impl_blis.hpp"

namespace tblis
{
namespace impl
{

enum impl_t
{
    REFERENCE  = 0,
    BLAS_BASED = 1,
    BLIS_BASED = 2
};

extern impl_t impl_type;

template <typename T>
int tensor_mult_impl(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                              const const_tensor_view<T>& B, const std::string& idx_B,
                     T  beta, const       tensor_view<T>& C, const std::string& idx_C)
{
    switch (impl_type)
    {
        case REFERENCE:
            return tensor_mult_reference(alpha, A, idx_A,
                                                B, idx_B,
                                          beta, C, idx_C);
            break;
        case BLAS_BASED:
            return tensor_mult_blas(alpha, A, idx_A,
                                           B, idx_B,
                                     beta, C, idx_C);
            break;
        case BLIS_BASED:
            abort();
            break;
    }
    return 0;
}

template <typename T>
int tensor_contract_impl(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                  const const_tensor_view<T>& B, const std::string& idx_B,
                         T  beta, const       tensor_view<T>& C, const std::string& idx_C)
{
    switch (impl_type)
    {
        case REFERENCE:
            return tensor_contract_reference(alpha, A, idx_A,
                                                    B, idx_B,
                                              beta, C, idx_C);
            break;
        case BLAS_BASED:
            return tensor_contract_blas(alpha, A, idx_A,
                                               B, idx_B,
                                         beta, C, idx_C);
            break;
        case BLIS_BASED:
            return tensor_contract_blis(alpha, A, idx_A,
                                               B, idx_B,
                                         beta, C, idx_C);
            break;
    }
    return 0;
}

template <typename T>
int tensor_weight_impl(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                const const_tensor_view<T>& B, const std::string& idx_B,
                       T  beta, const       tensor_view<T>& C, const std::string& idx_C)
{
    switch (impl_type)
    {
        case REFERENCE:
            return tensor_weight_reference(alpha, A, idx_A,
                                                  B, idx_B,
                                            beta, C, idx_C);
            break;
        case BLAS_BASED:
            return tensor_weight_blas(alpha, A, idx_A,
                                             B, idx_B,
                                       beta, C, idx_C);
            break;
        case BLIS_BASED:
            abort();
            break;
    }
    return 0;
}

template <typename T>
int tensor_outer_prod_impl(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                    const const_tensor_view<T>& B, const std::string& idx_B,
                           T  beta, const       tensor_view<T>& C, const std::string& idx_C)
{
    switch (impl_type)
    {
        case REFERENCE:
            return tensor_outer_prod_reference(alpha, A, idx_A,
                                                      B, idx_B,
                                                beta, C, idx_C);
            break;
        case BLAS_BASED:
            return tensor_outer_prod_blas(alpha, A, idx_A,
                                                 B, idx_B,
                                           beta, C, idx_C);
            break;
        case BLIS_BASED:
            abort();
            break;
    }
    return 0;
}

template <typename T>
int tensor_sum_impl(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                    T  beta, const       tensor_view<T>& B, const std::string& idx_B)
{
    switch (impl_type)
    {
        case REFERENCE:
        case BLAS_BASED:
            return tensor_sum_reference(alpha, A, idx_A,
                                         beta, B, idx_B);
            break;
        case BLIS_BASED:
            abort();
            break;
    }
    return 0;
}

template <typename T>
int tensor_trace_impl(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                      T  beta, const       tensor_view<T>& B, const std::string& idx_B)
{
    switch (impl_type)
    {
        case REFERENCE:
        case BLAS_BASED:
            return tensor_trace_reference(alpha, A, idx_A,
                                           beta, B, idx_B);
            break;
        case BLIS_BASED:
            abort();
            break;
    }
    return 0;
}

template <typename T>
int tensor_replicate_impl(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                          T  beta, const       tensor_view<T>& B, const std::string& idx_B)
{
    switch (impl_type)
    {
        case REFERENCE:
        case BLAS_BASED:
            return tensor_replicate_reference(alpha, A, idx_A,
                                               beta, B, idx_B);
            break;
        case BLIS_BASED:
            abort();
            break;
    }
    return 0;
}

template <typename T>
int tensor_transpose_impl(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                          T  beta, const       tensor_view<T>& B, const std::string& idx_B)
{
    switch (impl_type)
    {
        case REFERENCE:
        case BLAS_BASED:
            return tensor_transpose_reference(alpha, A, idx_A,
                                               beta, B, idx_B);
            break;
        case BLIS_BASED:
            abort();
            break;
    }
    return 0;
}

template <typename T>
int tensor_dot_impl(const const_tensor_view<T>& A, const std::string& idx_A,
                    const const_tensor_view<T>& B, const std::string& idx_B, T& val)
{
    switch (impl_type)
    {
        case REFERENCE:
        case BLAS_BASED:
            return tensor_dot_reference(A, idx_A,
                                        B, idx_B, val);
            break;
        case BLIS_BASED:
            abort();
            break;
    }
    return 0;
}

template <typename T>
int tensor_scale_impl(T alpha, const tensor_view<T>& A, const std::string& idx_A)
{
    switch (impl_type)
    {
        case REFERENCE:
        case BLAS_BASED:
            return tensor_scale_reference(alpha, A, idx_A);
            break;
        case BLIS_BASED:
            abort();
            break;
    }
    return 0;
}

template <typename T>
int tensor_reduce_impl(reduce_t op, const const_tensor_view<T>& A, const std::string& idx_A, T& val, stride_type& idx)
{
    switch (impl_type)
    {
        case REFERENCE:
        case BLAS_BASED:
            return tensor_reduce_reference(op, A, idx_A, val, idx);
            break;
        case BLIS_BASED:
            abort();
            break;
    }
    return 0;
}

}
}

#endif
