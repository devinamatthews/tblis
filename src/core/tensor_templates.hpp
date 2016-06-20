#ifndef _TBLIS_TEMPLATES_HPP_
#define _TBLIS_TEMPLATES_HPP_

#include "tblis.hpp"

namespace tblis
{

namespace detail
{
    template <typename... Args> struct has_member;

    template <typename T, typename=void>
    struct pointer_type;

    template <typename T>
    struct pointer_type<T, stl_ext::enable_if_t<std::is_pointer<T>::value>>
    {
        typedef stl_ext::remove_pointer_t<T> type;
    };

    template <typename T>
    struct pointer_type<T, stl_ext::conditional_t<false,
        has_member<decltype(std::declval<T>().data()),
                   decltype(std::declval<T>().size())>,void>>
    {
        typedef stl_ext::remove_pointer_t<decltype(std::declval<T>().data())> type;
    };

    template <typename T>
    using pointer_type_t = typename pointer_type<T>::type;

    template <typename... Args> struct check_template_types;

    template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
              typename U>
    struct check_template_types<T, A_ptr, A_len, A_stride, A_idx, U>
    {
        typedef stl_ext::enable_if<(std::is_same<float,T>::value ||
                                    std::is_same<double,T>::value ||
                                    std::is_same<scomplex,T>::value ||
                                    std::is_same<dcomplex,T>::value) &&
                                   std::is_same<pointer_type_t<A_ptr>,T>::value &&
                                   std::is_integral<pointer_type_t<A_len>>::value &&
                                   std::is_integral<pointer_type_t<A_stride>>::value &&
                                   std::is_integral<pointer_type_t<A_idx>>::value,
                                   U> type;
    };

    template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                          typename B_ptr, typename B_len, typename B_stride, typename B_idx,
              typename U>
    struct check_template_types<T, A_ptr, A_len, A_stride, A_idx,
                                   B_ptr, B_len, B_stride, B_idx, U>
    {
        typedef stl_ext::enable_if<(std::is_same<float,T>::value ||
                                    std::is_same<double,T>::value ||
                                    std::is_same<scomplex,T>::value ||
                                    std::is_same<dcomplex,T>::value) &&
                                   std::is_same<pointer_type_t<A_ptr>,T>::value &&
                                   std::is_same<pointer_type_t<B_ptr>,T>::value &&
                                   std::is_integral<pointer_type_t<A_len>>::value &&
                                   std::is_integral<pointer_type_t<B_len>>::value &&
                                   std::is_integral<pointer_type_t<A_stride>>::value &&
                                   std::is_integral<pointer_type_t<B_stride>>::value &&
                                   std::is_integral<pointer_type_t<A_idx>>::value &&
                                   std::is_integral<pointer_type_t<B_idx>>::value,
                                   U> type;
    };

    template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                          typename B_ptr, typename B_len, typename B_stride, typename B_idx,
                          typename C_ptr, typename C_len, typename C_stride, typename C_idx,
              typename U>
    struct check_template_types<T, A_ptr, A_len, A_stride, A_idx,
                                   B_ptr, B_len, B_stride, B_idx,
                                   C_ptr, C_len, C_stride, C_idx, U>
    {
        typedef stl_ext::enable_if<(std::is_same<float,T>::value ||
                                    std::is_same<double,T>::value ||
                                    std::is_same<scomplex,T>::value ||
                                    std::is_same<dcomplex,T>::value) &&
                                   std::is_same<pointer_type_t<A_ptr>,T>::value &&
                                   std::is_same<pointer_type_t<B_ptr>,T>::value &&
                                   std::is_same<pointer_type_t<C_ptr>,T>::value &&
                                   std::is_integral<pointer_type_t<A_len>>::value &&
                                   std::is_integral<pointer_type_t<B_len>>::value &&
                                   std::is_integral<pointer_type_t<C_len>>::value &&
                                   std::is_integral<pointer_type_t<A_stride>>::value &&
                                   std::is_integral<pointer_type_t<B_stride>>::value &&
                                   std::is_integral<pointer_type_t<C_stride>>::value &&
                                   std::is_integral<pointer_type_t<A_idx>>::value &&
                                   std::is_integral<pointer_type_t<B_idx>>::value &&
                                   std::is_integral<pointer_type_t<C_idx>>::value,
                                   U> type;
    };

    template <typename... Args>
    using check_template_types_t = typename check_template_types<Args...>::type;

    template <typename Len>
    stl_ext::enable_if_t<std::is_pointer<Len>::value,std::vector<dim_t>>
    make_len(unsigned ndim, const Len& x)
    {
        return {x, x+ndim};
    }

    template <typename Len>
    stl_ext::enable_if_t<!std::is_pointer<Len>::value,std::vector<dim_t>>
    make_len(unsigned ndim, const Len& x)
    {
        assert(x.size() == ndim);
        return {x.data(), x.data()+ndim};
    }

    template <typename Stride>
    stl_ext::enable_if_t<std::is_pointer<Stride>::value,std::vector<inc_t>>
    make_stride(unsigned ndim, const Stride& x)
    {
        return {x, x+ndim};
    }

    template <typename Stride>
    stl_ext::enable_if_t<!std::is_pointer<Stride>::value,std::vector<inc_t>>
    make_stride(unsigned ndim, const Stride& x)
    {
        assert(x.size() == ndim);
        return {x.data(), x.data()+ndim};
    }

    template <typename Idx>
    stl_ext::enable_if_t<std::is_pointer<Idx>::value,std::string>
    make_idx(unsigned ndim, const Idx& x)
    {
        return {x, x+ndim};
    }

    template <typename Idx>
    stl_ext::enable_if_t<!std::is_pointer<Idx>::value,std::string>
    make_idx(unsigned ndim, const Idx& x)
    {
        assert(x.size() == ndim);
        return {x.data(), x.data()+ndim};
    }
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx,
                      typename C_ptr, typename C_len, typename C_stride, typename C_idx>
detail::check_template_types_t<T, A_ptr, A_len, A_stride, A_idx,
                                  B_ptr, B_len, B_stride, B_idx,
                                  C_ptr, C_len, C_stride, C_idx, int>
tensor_mult(T alpha, const A_ptr& A, unsigned ndim_A, const A_len& len_A, const A_stride& stride_A, const A_idx& idx_A,
                     const B_ptr& B, unsigned ndim_B, const B_len& len_B, const B_stride& stride_B, const B_idx& idx_B,
            T  beta,       C_ptr& C, unsigned ndim_C, const C_len& len_C, const C_stride& stride_C, const C_idx& idx_C)
{
    const_tensor_view<T> A_(make_len(ndim_A, len_A), A, make_stride(ndim_A, stride_A));
    const_tensor_view<T> B_(make_len(ndim_B, len_B), B, make_stride(ndim_B, stride_B));
          tensor_view<T> C_(make_len(ndim_C, len_C), C, make_stride(ndim_C, stride_C));

    return tensor_mult(alpha, A_, make_idx(ndim_A, idx_A),
                              B_, make_idx(ndim_B, idx_B),
                        beta, C_, make_idx(ndim_C, idx_C));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx,
                      typename C_ptr, typename C_len, typename C_stride, typename C_idx>
detail::check_template_types_t<T, A_ptr, A_len, A_stride, A_idx,
                                  B_ptr, B_len, B_stride, B_idx,
                                  C_ptr, C_len, C_stride, C_idx, int>
tensor_contract(T alpha, const A_ptr& A, unsigned ndim_A, const A_len& len_A, const A_stride& stride_A, const A_idx& idx_A,
                     const B_ptr& B, unsigned ndim_B, const B_len& len_B, const B_stride& stride_B, const B_idx& idx_B,
            T  beta,       C_ptr& C, unsigned ndim_C, const C_len& len_C, const C_stride& stride_C, const C_idx& idx_C)
{
    const_tensor_view<T> A_(make_len(ndim_A, len_A), A, make_stride(ndim_A, stride_A));
    const_tensor_view<T> B_(make_len(ndim_B, len_B), B, make_stride(ndim_B, stride_B));
          tensor_view<T> C_(make_len(ndim_C, len_C), C, make_stride(ndim_C, stride_C));

    return tensor_contract(alpha, A_, make_idx(ndim_A, idx_A),
                                  B_, make_idx(ndim_B, idx_B),
                            beta, C_, make_idx(ndim_C, idx_C));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx,
                      typename C_ptr, typename C_len, typename C_stride, typename C_idx>
detail::check_template_types_t<T, A_ptr, A_len, A_stride, A_idx,
                                  B_ptr, B_len, B_stride, B_idx,
                                  C_ptr, C_len, C_stride, C_idx, int>
tensor_weight(T alpha, const A_ptr& A, unsigned ndim_A, const A_len& len_A, const A_stride& stride_A, const A_idx& idx_A,
                     const B_ptr& B, unsigned ndim_B, const B_len& len_B, const B_stride& stride_B, const B_idx& idx_B,
            T  beta,       C_ptr& C, unsigned ndim_C, const C_len& len_C, const C_stride& stride_C, const C_idx& idx_C)
{
    const_tensor_view<T> A_(make_len(ndim_A, len_A), A, make_stride(ndim_A, stride_A));
    const_tensor_view<T> B_(make_len(ndim_B, len_B), B, make_stride(ndim_B, stride_B));
          tensor_view<T> C_(make_len(ndim_C, len_C), C, make_stride(ndim_C, stride_C));

    return tensor_weight(alpha, A_, make_idx(ndim_A, idx_A),
                                B_, make_idx(ndim_B, idx_B),
                          beta, C_, make_idx(ndim_C, idx_C));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx,
                      typename C_ptr, typename C_len, typename C_stride, typename C_idx>
detail::check_template_types_t<T, A_ptr, A_len, A_stride, A_idx,
                                  B_ptr, B_len, B_stride, B_idx,
                                  C_ptr, C_len, C_stride, C_idx, int>
tensor_outer_prod(T alpha, const A_ptr& A, unsigned ndim_A, const A_len& len_A, const A_stride& stride_A, const A_idx& idx_A,
                     const B_ptr& B, unsigned ndim_B, const B_len& len_B, const B_stride& stride_B, const B_idx& idx_B,
            T  beta,       C_ptr& C, unsigned ndim_C, const C_len& len_C, const C_stride& stride_C, const C_idx& idx_C)
{
    const_tensor_view<T> A_(make_len(ndim_A, len_A), A, make_stride(ndim_A, stride_A));
    const_tensor_view<T> B_(make_len(ndim_B, len_B), B, make_stride(ndim_B, stride_B));
          tensor_view<T> C_(make_len(ndim_C, len_C), C, make_stride(ndim_C, stride_C));

    return tensor_outer_prod(alpha, A_, make_idx(ndim_A, idx_A),
                                    B_, make_idx(ndim_B, idx_B),
                              beta, C_, make_idx(ndim_C, idx_C));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx>
detail::check_template_types_t<T, A_ptr, A_len, A_stride, A_idx,
                                  B_ptr, B_len, B_stride, B_idx, int>
tensor_sum(T alpha, const A_ptr& A, unsigned ndim_A, const A_len& len_A, const A_stride& stride_A, const A_idx& idx_A,
           T  beta, const B_ptr& B, unsigned ndim_B, const B_len& len_B, const B_stride& stride_B, const B_idx& idx_B)
{
    const_tensor_view<T> A_(make_len(ndim_A, len_A), A, make_stride(ndim_A, stride_A));
          tensor_view<T> B_(make_len(ndim_B, len_B), B, make_stride(ndim_B, stride_B));

    return tensor_sum(alpha, A_, make_idx(ndim_A, idx_A),
                       beta, B_, make_idx(ndim_B, idx_B));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx>
detail::check_template_types_t<T, A_ptr, A_len, A_stride, A_idx,
                                  B_ptr, B_len, B_stride, B_idx, int>
tensor_trace(T alpha, const A_ptr& A, unsigned ndim_A, const A_len& len_A, const A_stride& stride_A, const A_idx& idx_A,
           T  beta, const B_ptr& B, unsigned ndim_B, const B_len& len_B, const B_stride& stride_B, const B_idx& idx_B)
{
    const_tensor_view<T> A_(make_len(ndim_A, len_A), A, make_stride(ndim_A, stride_A));
          tensor_view<T> B_(make_len(ndim_B, len_B), B, make_stride(ndim_B, stride_B));

    return tensor_trace(alpha, A_, make_idx(ndim_A, idx_A),
                         beta, B_, make_idx(ndim_B, idx_B));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx>
detail::check_template_types_t<T, A_ptr, A_len, A_stride, A_idx,
                                  B_ptr, B_len, B_stride, B_idx, int>
tensor_replicate(T alpha, const A_ptr& A, unsigned ndim_A, const A_len& len_A, const A_stride& stride_A, const A_idx& idx_A,
           T  beta, const B_ptr& B, unsigned ndim_B, const B_len& len_B, const B_stride& stride_B, const B_idx& idx_B)
{
    const_tensor_view<T> A_(make_len(ndim_A, len_A), A, make_stride(ndim_A, stride_A));
          tensor_view<T> B_(make_len(ndim_B, len_B), B, make_stride(ndim_B, stride_B));

    return tensor_replicate(alpha, A_, make_idx(ndim_A, idx_A),
                             beta, B_, make_idx(ndim_B, idx_B));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx>
detail::check_template_types_t<T, A_ptr, A_len, A_stride, A_idx,
                                  B_ptr, B_len, B_stride, B_idx, int>
tensor_transpose(T alpha, const A_ptr& A, unsigned ndim_A, const A_len& len_A, const A_stride& stride_A, const A_idx& idx_A,
           T  beta, const B_ptr& B, unsigned ndim_B, const B_len& len_B, const B_stride& stride_B, const B_idx& idx_B)
{
    const_tensor_view<T> A_(make_len(ndim_A, len_A), A, make_stride(ndim_A, stride_A));
          tensor_view<T> B_(make_len(ndim_B, len_B), B, make_stride(ndim_B, stride_B));

    return tensor_transpose(alpha, A_, make_idx(ndim_A, idx_A),
                             beta, B_, make_idx(ndim_B, idx_B));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx>
detail::check_template_types_t<T, A_ptr, A_len, A_stride, A_idx,
                                  B_ptr, B_len, B_stride, B_idx, T>
tensor_dot(T alpha, const A_ptr& A, unsigned ndim_A, const A_len& len_A, const A_stride& stride_A, const A_idx& idx_A,
           T  beta, const B_ptr& B, unsigned ndim_B, const B_len& len_B, const B_stride& stride_B, const B_idx& idx_B)
{
    const_tensor_view<T> A_(make_len(ndim_A, len_A), A, make_stride(ndim_A, stride_A));
    const_tensor_view<T> B_(make_len(ndim_B, len_B), B, make_stride(ndim_B, stride_B));

    return tensor_dot(A_, make_idx(ndim_A, idx_A),
                      B_, make_idx(ndim_B, idx_B));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx>
detail::check_template_types_t<T, A_ptr, A_len, A_stride, A_idx,
                                  B_ptr, B_len, B_stride, B_idx, int>
tensor_dot(T alpha, const A_ptr& A, unsigned ndim_A, const A_len& len_A, const A_stride& stride_A, const A_idx& idx_A,
           T  beta, const B_ptr& B, unsigned ndim_B, const B_len& len_B, const B_stride& stride_B, const B_idx& idx_B,
           T& val)
{
    const_tensor_view<T> A_(make_len(ndim_A, len_A), A, make_stride(ndim_A, stride_A));
    const_tensor_view<T> B_(make_len(ndim_B, len_B), B, make_stride(ndim_B, stride_B));

    return tensor_dot(A_, make_idx(ndim_A, idx_A),
                      B_, make_idx(ndim_B, idx_B), val);
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx>
detail::check_template_types_t<T, A_ptr, A_len, A_stride, A_idx, int>
tensor_scale(T alpha, A_ptr& A, unsigned ndim_A, const A_len& len_A, const A_stride& stride_A, const A_idx& idx_A)
{
    tensor_view<T> A_(make_len(ndim_A, len_A), A, make_stride(ndim_A, stride_A));
    return tensor_scale(alpha, A_, make_idx(ndim_A, idx_A));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx>
detail::check_template_types_t<T, A_ptr, A_len, A_stride, A_idx, std::pair<T,inc_t>>
tensor_reduce(reduce_t op, const A_ptr& A, unsigned ndim_A, const A_len& len_A, const A_stride& stride_A, const A_idx& idx_A)
{
    const_tensor_view<T> A_(make_len(ndim_A, len_A), A, make_stride(ndim_A, stride_A));
    return tensor_reduce(op, A_, make_idx(ndim_A, idx_A));
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx>
detail::check_template_types_t<T, A_ptr, A_len, A_stride, A_idx, int>
tensor_reduce(reduce_t op, const A_ptr& A, unsigned ndim_A, const A_len& len_A, const A_stride& stride_A, const A_idx& idx_A,
              inc_t& idx)
{
    const_tensor_view<T> A_(make_len(ndim_A, len_A), A, make_stride(ndim_A, stride_A));
    return tensor_reduce(op, A_, make_idx(ndim_A, idx_A), idx);
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx>
detail::check_template_types_t<T, A_ptr, A_len, A_stride, A_idx, int>
tensor_reduce(reduce_t op, const A_ptr& A, unsigned ndim_A, const A_len& len_A, const A_stride& stride_A, const A_idx& idx_A,
              T& val)
{
    const_tensor_view<T> A_(make_len(ndim_A, len_A), A, make_stride(ndim_A, stride_A));
    return tensor_reduce(op, A_, make_idx(ndim_A, idx_A), val);
}

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx>
detail::check_template_types_t<T, A_ptr, A_len, A_stride, A_idx, int>
tensor_reduce(reduce_t op, const A_ptr& A, unsigned ndim_A, const A_len& len_A, const A_stride& stride_A, const A_idx& idx_A,
              T& val, inc_t& idx)
{
    const_tensor_view<T> A_(make_len(ndim_A, len_A), A, make_stride(ndim_A, stride_A));
    return tensor_reduce(op, A_, make_idx(ndim_A, idx_A), val, idx);
}

}

#endif
