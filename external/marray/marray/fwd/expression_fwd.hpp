#ifndef MARRAY_EXPRESSION_FWD_HPP
#define MARRAY_EXPRESSION_FWD_HPP

#include <type_traits>

namespace MArray
{

template <typename Expr, typename=void> struct is_expression;

template <typename T> struct is_scalar;

template <typename Expr, typename=void> struct is_array_expression;

template <typename Expr, typename=void> struct is_unary_expression;

template <typename Expr, typename=void> struct is_binary_expression;

template <typename Array, typename=void> struct is_marray;

template <typename Expr> struct is_expression_arg;

template <typename Expr> struct is_expression_arg_or_scalar;

template <typename Expr, typename=void> struct expr_result_type;

template <typename Expr> using expr_result_type_t = typename expr_result_type<Expr>::type;

template <typename Array, typename Expr>
std::enable_if_t<(is_array_expression<std::decay_t<Array>>::value ||
                     is_marray<std::decay_t<Array>>::value) &&
                    is_expression_arg_or_scalar<std::decay_t<Expr>>::value>
assign_expr(Array&& array_, Expr&& expr_);

}

#endif //MARRAY_EXPRESSION_FWD_HPP
