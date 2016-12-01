#ifndef _TBLIS_CONFIG_BUILDER_HPP_
#define _TBLIS_CONFIG_BUILDER_HPP_

#include "configs.hpp"

namespace tblis
{

template <typename T> using pack_nn_mr_ukr_t = pack_nn_ukr_t<T>;
template <typename T> using pack_nn_nr_ukr_t = pack_nn_ukr_t<T>;
template <typename T> using pack_sn_mr_ukr_t = pack_sn_ukr_t<T>;
template <typename T> using pack_sn_nr_ukr_t = pack_sn_ukr_t<T>;
template <typename T> using pack_ns_mr_ukr_t = pack_ns_ukr_t<T>;
template <typename T> using pack_ns_nr_ukr_t = pack_ns_ukr_t<T>;
template <typename T> using pack_ss_mr_ukr_t = pack_ss_ukr_t<T>;
template <typename T> using pack_ss_nr_ukr_t = pack_ss_ukr_t<T>;
template <typename T> using pack_nb_mr_ukr_t = pack_nb_ukr_t<T>;
template <typename T> using pack_nb_nr_ukr_t = pack_nb_ukr_t<T>;
template <typename T> using pack_sb_mr_ukr_t = pack_sb_ukr_t<T>;
template <typename T> using pack_sb_nr_ukr_t = pack_sb_ukr_t<T>;

template <typename T, len_type S, len_type D, len_type C, len_type Z>
struct simple_blocksize;

template <len_type S, len_type D, len_type C, len_type Z>
struct simple_blocksize<float, S, D, C, Z>
{
    static constexpr len_type def = S;
};

template <len_type S, len_type D, len_type C, len_type Z>
struct simple_blocksize<double, S, D, C, Z>
{
    static constexpr len_type def = D;
};

template <len_type S, len_type D, len_type C, len_type Z>
struct simple_blocksize<scomplex, S, D, C, Z>
{
    static constexpr len_type def = C;
};

template <len_type S, len_type D, len_type C, len_type Z>
struct simple_blocksize<dcomplex, S, D, C, Z>
{
    static constexpr len_type def = Z;
};

#define TBLIS_DEFAULT_VALUE(name, type, source, val, def) \
    template <typename U> \
    static std::integral_constant<type, U::val> _##name##_helper(U*); \
    template <typename U> \
    static std::integral_constant<type,    def> _##name##_helper(...); \
    static constexpr type name = decltype(_##name##_helper<source>(static_cast<source*>(nullptr)))::value;

/*
 * The cast on the first overload is very important; otherwise the
 * compiler silently chooses the default kernel because it wants a
 * direct pointer to an extern function.
 */
#define TBLIS_DEFAULT_VALUE_T(name, tp, source, val, def) \
    template <typename U, typename T> \
    static std::integral_constant<tp, static_cast<tp>(U::template val<T>::value)> _##name##_helper(U*); \
    template <typename U, typename T> \
    static std::integral_constant<tp, def> _##name##_helper(...); \
    template <typename T> \
    struct name : decltype(_##name##_helper<source, T>(static_cast<source*>(nullptr))) {};

template <typename BS,
          typename BS_Ref,
          typename BS_Iota=BS>
struct blocksize_traits
{
    struct type
    {
        TBLIS_DEFAULT_VALUE(      def, len_type,      BS,    def, BS_Ref::def)
        TBLIS_DEFAULT_VALUE(      max, len_type,      BS,    max,         def)
        TBLIS_DEFAULT_VALUE(   extent, len_type,      BS, extent,         def)
        /*
         * If BS_Iota == BS, then it may not have a def member type, so we need
         * a second layer of defaulting (in this case iota will come out the
         * same as def in the end). If BS_Iota is different then it must have a
         * def member type or this default doesn't make sense.
         */
        TBLIS_DEFAULT_VALUE(_iota_def, len_type, BS_Iota,    def, BS_Ref::def)
        TBLIS_DEFAULT_VALUE(     iota, len_type,      BS,   iota,   _iota_def)
    };
};

template <typename Config>
struct config_traits : config
{
    TBLIS_DEFAULT_VALUE_T(   add_ukr,    add_ukr_t<T>, Config,    add_ukr,    &add_ukr_def<T>)
    TBLIS_DEFAULT_VALUE_T(  copy_ukr,   copy_ukr_t<T>, Config,   copy_ukr,   &copy_ukr_def<T>)
    TBLIS_DEFAULT_VALUE_T(   dot_ukr,    dot_ukr_t<T>, Config,    dot_ukr,    &dot_ukr_def<T>)
    TBLIS_DEFAULT_VALUE_T(reduce_ukr, reduce_ukr_t<T>, Config, reduce_ukr, &reduce_ukr_def<T>)
    TBLIS_DEFAULT_VALUE_T( scale_ukr,  scale_ukr_t<T>, Config,  scale_ukr,  &scale_ukr_def<T>)
    TBLIS_DEFAULT_VALUE_T(   set_ukr,    set_ukr_t<T>, Config,    set_ukr,    &set_ukr_def<T>)

    template <typename T> struct _default_trans_mr : simple_blocksize<T, 8, 4, 4, 4> {};
    template <typename T> struct _default_trans_nr : simple_blocksize<T, 4, 4, 4, 2> {};

    template <typename T> struct trans_mr : blocksize_traits<
        typename Config::template trans_mr<T>, _default_trans_mr<T>>::type {};
    template <typename T> struct trans_nr : blocksize_traits<
        typename Config::template trans_nr<T>, _default_trans_nr<T>>::type {};

    TBLIS_DEFAULT_VALUE_T(  trans_add_ukr,  trans_add_ukr_t<T>, Config,   trans_add_ukr,  (&trans_add_ukr_def<config_traits,T>))
    TBLIS_DEFAULT_VALUE_T( trans_copy_ukr, trans_copy_ukr_t<T>, Config,  trans_copy_ukr, (&trans_copy_ukr_def<config_traits,T>))
    TBLIS_DEFAULT_VALUE_T(trans_row_major,                bool, Config, trans_row_major,                                  false)

    template <typename T> struct _default_gemm_mr : simple_blocksize<T, 8, 4, 4, 2> {};
    template <typename T> struct _default_gemm_nr : simple_blocksize<T, 4, 4, 2, 2> {};
    template <typename T> struct _default_gemm_kr : simple_blocksize<T, 4, 2, 2, 1> {};

    template <typename T> struct gemm_mr : blocksize_traits<
        typename Config::template gemm_mr<T>, _default_gemm_mr<T>>::type {};
    template <typename T> struct gemm_nr : blocksize_traits<
        typename Config::template gemm_nr<T>, _default_gemm_nr<T>>::type {};
    template <typename T> struct gemm_kr : blocksize_traits<
        typename Config::template gemm_kr<T>, _default_gemm_kr<T>>::type {};

    template <typename T> struct _default_gemm_mc : simple_blocksize<T,  512,  256,  256,  128> {};
    template <typename T> struct _default_gemm_nc : simple_blocksize<T, 4096, 4096, 4096, 4096> {};
    template <typename T> struct _default_gemm_kc : simple_blocksize<T,  256,  256,  256,  256> {};

    template <typename T> struct gemm_mc : blocksize_traits<
        typename Config::template gemm_mc<T>, _default_gemm_mc<T>, gemm_mr<T>>::type {};
    template <typename T> struct gemm_nc : blocksize_traits<
        typename Config::template gemm_nc<T>, _default_gemm_nc<T>, gemm_nr<T>>::type {};
    template <typename T> struct gemm_kc : blocksize_traits<
        typename Config::template gemm_kc<T>, _default_gemm_kc<T>, gemm_kr<T>>::type {};

    TBLIS_DEFAULT_VALUE_T(      gemm_ukr, gemm_ukr_t<T>, Config,       gemm_ukr, (&gemm_ukr_def<config_traits,T>))
    TBLIS_DEFAULT_VALUE_T(gemm_row_major,          bool, Config, gemm_row_major,                            false)

    TBLIS_DEFAULT_VALUE_T(pack_nn_mr_ukr, pack_nn_ukr_t<T>, Config, pack_nn_mr_ukr, (pack_nn_ukr_def<config_traits,T,matrix_constants::MAT_A>))
    TBLIS_DEFAULT_VALUE_T(pack_nn_nr_ukr, pack_nn_ukr_t<T>, Config, pack_nn_nr_ukr, (pack_nn_ukr_def<config_traits,T,matrix_constants::MAT_B>))
    TBLIS_DEFAULT_VALUE_T(pack_sn_mr_ukr, pack_sn_ukr_t<T>, Config, pack_sn_mr_ukr, (pack_sn_ukr_def<config_traits,T,matrix_constants::MAT_A>))
    TBLIS_DEFAULT_VALUE_T(pack_sn_nr_ukr, pack_sn_ukr_t<T>, Config, pack_sn_nr_ukr, (pack_sn_ukr_def<config_traits,T,matrix_constants::MAT_B>))
    TBLIS_DEFAULT_VALUE_T(pack_ns_mr_ukr, pack_ns_ukr_t<T>, Config, pack_ns_mr_ukr, (pack_ns_ukr_def<config_traits,T,matrix_constants::MAT_A>))
    TBLIS_DEFAULT_VALUE_T(pack_ns_nr_ukr, pack_ns_ukr_t<T>, Config, pack_ns_nr_ukr, (pack_ns_ukr_def<config_traits,T,matrix_constants::MAT_B>))
    TBLIS_DEFAULT_VALUE_T(pack_ss_mr_ukr, pack_ss_ukr_t<T>, Config, pack_ss_mr_ukr, (pack_ss_ukr_def<config_traits,T,matrix_constants::MAT_A>))
    TBLIS_DEFAULT_VALUE_T(pack_ss_nr_ukr, pack_ss_ukr_t<T>, Config, pack_ss_nr_ukr, (pack_ss_ukr_def<config_traits,T,matrix_constants::MAT_B>))
    TBLIS_DEFAULT_VALUE_T(pack_nb_mr_ukr, pack_nb_ukr_t<T>, Config, pack_nb_mr_ukr, (pack_nb_ukr_def<config_traits,T,matrix_constants::MAT_A>))
    TBLIS_DEFAULT_VALUE_T(pack_nb_nr_ukr, pack_nb_ukr_t<T>, Config, pack_nb_nr_ukr, (pack_nb_ukr_def<config_traits,T,matrix_constants::MAT_B>))
    TBLIS_DEFAULT_VALUE_T(pack_sb_mr_ukr, pack_sb_ukr_t<T>, Config, pack_sb_mr_ukr, (pack_sb_ukr_def<config_traits,T,matrix_constants::MAT_A>))
    TBLIS_DEFAULT_VALUE_T(pack_sb_nr_ukr, pack_sb_ukr_t<T>, Config, pack_sb_nr_ukr, (pack_sb_ukr_def<config_traits,T,matrix_constants::MAT_B>))

    static constexpr check_fn_t check = &Config::check;

    static constexpr const char* name = Config::name;

    static const config& instance()
    {
        return Config::instance();
    }

    config_traits() : config(*this) {}
};

#define TBLIS_CONFIG(cfg) \
extern config cfg##_config_instance; \
struct cfg##_config_ \
{ \
    template <typename T> struct    add_ukr {}; \
    template <typename T> struct   copy_ukr {}; \
    template <typename T> struct    dot_ukr {}; \
    template <typename T> struct reduce_ukr {}; \
    template <typename T> struct  scale_ukr {}; \
    template <typename T> struct    set_ukr {}; \
    \
    template <typename T> struct trans_mr {}; \
    template <typename T> struct trans_nr {}; \
    \
    template <typename T> struct trans_add_ukr {}; \
    template <typename T> struct trans_copy_ukr {}; \
    template <typename T> struct trans_row_major {}; \
    \
    template <typename T> struct gemm_mr {}; \
    template <typename T> struct gemm_nr {}; \
    template <typename T> struct gemm_kr {}; \
    template <typename T> struct gemm_mc {}; \
    template <typename T> struct gemm_nc {}; \
    template <typename T> struct gemm_kc {}; \
    \
    template <typename T> struct gemm_ukr {}; \
    template <typename T> struct gemm_row_major {}; \
    \
    template <typename T> struct pack_nn_mr_ukr {}; \
    template <typename T> struct pack_nn_nr_ukr {}; \
    template <typename T> struct pack_sn_mr_ukr {}; \
    template <typename T> struct pack_sn_nr_ukr {}; \
    template <typename T> struct pack_ns_mr_ukr {}; \
    template <typename T> struct pack_ns_nr_ukr {}; \
    template <typename T> struct pack_ss_mr_ukr {}; \
    template <typename T> struct pack_ss_nr_ukr {}; \
    template <typename T> struct pack_nb_mr_ukr {}; \
    template <typename T> struct pack_nb_nr_ukr {}; \
    template <typename T> struct pack_sb_mr_ukr {}; \
    template <typename T> struct pack_sb_nr_ukr {}; \
    \
    static int check(); \
    static constexpr const char* name = #cfg; \
    \
    static const config& instance() { return cfg##_config_instance; } \
}; \
typedef config_traits<cfg##_config_> cfg##_config;

#define TBLIS_CONFIG_X_1(config, X, T, type1, name1, value1) \
template <> struct config##_config_::X<T> \
{ \
    static constexpr type1 name1 = value1; \
};

#define TBLIS_CONFIG_X_2(config, X, T, type1, name1, value1, \
                                       type2, name2, value2) \
template <> struct config##_config_::X<T> \
{ \
    static constexpr type1 name1 = value1; \
    static constexpr type2 name2 = value2; \
};

#define TBLIS_CONFIG_UKR(config, T, op, func) \
TBLIS_CONFIG_X_1(config, op##_ukr, T, op##_ukr_t<T>, value, func)

#define TBLIS_CONFIG_ROW_MAJOR(config, T, op) \
TBLIS_CONFIG_X_1(config, op##_row_major, T, bool, value, true)

#define TBLIS_CONFIG_BS_DEF(config, T, op, bs, _def) \
TBLIS_CONFIG_X_1(config, op##_##bs, T, len_type, def, _def)

#define TBLIS_CONFIG_BS_DEF_MAX(config, T, op, bs, _def, _max) \
TBLIS_CONFIG_X_2(config, op##_##bs, T, len_type, def, _def, \
                                       len_type, max, _max)

#define TBLIS_CONFIG_BS_DEF_EXTENT(config, T, op, bs, _def, _extent) \
TBLIS_CONFIG_X_2(config, op##_##bs, T, len_type, def, _def, \
                                       len_type, extent, _extent)

#define TBLIS_CONFIG_CHECK(config, chk) \
int config##_config_::check()

#define TBLIS_CONFIG_INSTANTIATE(cfg) \
extern config cfg##_config_instance; \
config cfg##_config_instance = config(cfg##_config());

}

#endif
