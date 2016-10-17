#ifndef _TBLIS_CONFIGS_HPP_
#define _TBLIS_CONFIGS_HPP_

#include "util/basic_types.h"

#include "kernels/1m/reduce.hpp"
#include "kernels/3m/gemm.hpp"

namespace tblis
{

namespace matrix_constants
{
    enum {MAT_A, MAT_B, MAT_C};
    enum {DIM_MR=0x0, DIM_NR=0x2, DIM_KR=0x4,
          DIM_MC=0x1, DIM_NC=0x3, DIM_KC=0x5,
          DIM_M=DIM_MR, DIM_N=DIM_NR, DIM_K=DIM_KR};
}

//
// Return priority if config can run on this HW, -1 otherwise
//
using check_fn_t = int (*)(void);

template <typename T> struct type_idx;

template <> struct type_idx<   float> { constexpr int value = 0; };
template <> struct type_idx<  double> { constexpr int value = 1; };
template <> struct type_idx<scomplex> { constexpr int value = 2; };
template <> struct type_idx<dcomplex> { constexpr int value = 3; };

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
private: \
    template <typename U> \
    static std::integral_constant<type,    U::val> _##name##_helper(U*); \
    template <typename U> \
    static std::integral_constant<type, (type)def> _##name##_helper(...); \
public: \
    static constexpr type name = decltype(_##name##_helper<source>((source*)0))::value;

#define TBLIS_DEFAULT_VALUE_T(name, type, source, val, def) \
private: \
    template <typename U, typename T> \
    static std::integral_constant<type, U::template val<T>::value> _##name##_helper(U*); \
    template <typename U, typename T> \
    static std::integral_constant<type,                 (type)def> _##name##_helper(...); \
public: \
    template <typename T> \
    using name = std::integral_constant<type,decltype(_##name##_helper<source, T>((source*)0))::value>;

struct blocksize
{
    len_type    _def[4];
    len_type    _max[4];
    len_type   _iota[4];
    len_type _extent[4];

    template <typename T> len_type    def() const { return    _def[type_idx<T>::value]; }
    template <typename T> len_type    max() const { return    _max[type_idx<T>::value]; }
    template <typename T> len_type   iota() const { return   _iota[type_idx<T>::value]; }
    template <typename T> len_type extent() const { return _extent[type_idx<T>::value]; }

    template <template <typename> class BS> blocksize(const BS&)
    : _def   {BS<float>::def,    BS<double>::def,    BS<double>::def,    BS<double>::def},
      _max   {BS<float>::max,    BS<double>::max,    BS<double>::max,    BS<double>::max},
      _iota  {BS<float>::iota,   BS<double>::iota,   BS<double>::iota,   BS<double>::iota},
      _extent{BS<float>::extent, BS<double>::extent, BS<double>::extent, BS<double>::extent} {}
};

template <template <typename> class BS,
          template <typename> class BS_Ref,
          template <typename> class BS_Iota=BS>
struct blocksize_traits
{
    template <typename T>
    struct type
    {
        TBLIS_DEFAULT_VALUE(      def, len_type,      BS<T>,    def, BS_Ref<T>::def);
        TBLIS_DEFAULT_VALUE(      max, len_type,      BS<T>,    max,            def);
        TBLIS_DEFAULT_VALUE(_iota_def, len_type, BS_Iota<T>,    def, BS_Ref<T>::def);
        TBLIS_DEFAULT_VALUE(     iota, len_type,      BS<T>,   iota,      _iota_def);
        TBLIS_DEFAULT_VALUE(   extent, len_type,      BS<T>, extent,            def);
    };
};

template <typename Config, int Dim> struct _BS;

template <typename Config>
struct _BS<Config, matrix_constants::DIM_MC>
{
    template <typename T> using type = typename Config::template MC<T>;
};

template <typename Config>
struct _BS<Config, matrix_constants::DIM_NC>
{
    template <typename T> using type = typename Config::template NC<T>;
};

template <typename Config>
struct _BS<Config, matrix_constants::DIM_KC>
{
    template <typename T> using type = typename Config::template KC<T>;
};

template <typename Config>
struct _BS<Config, matrix_constants::DIM_MR>
{
    template <typename T> using type = typename Config::template MR<T>;
};

template <typename Config>
struct _BS<Config, matrix_constants::DIM_NR>
{
    template <typename T> using type = typename Config::template NR<T>;
};

template <typename Config>
struct _BS<Config, matrix_constants::DIM_KR>
{
    template <typename T> using type = typename Config::template KR<T>;
};

struct config
{
    blocksize MR;
    blocksize NR;
    blocksize KR;
    blocksize MC;
    blocksize NC;
    blocksize KC;

    void (*_gemm_ukr[4])(void);

    template <typename T>
    void gemm_ukr(stride_type k,
                  const T* alpha,
                  const T* a, const T* b,
                  const T* beta,
                  T* c, stride_type rs_c, stride_type cs_c) const
    {
        gemm_ukr_t<T>(_gemm_ukr[type_idx<T>::value])
            (k, alpha, a, b, beta, c, rs_c, cs_c);
    }

    void (*_pack_nn_mr[4])(void);
    void (*_pack_nn_nr[4])(void);
    void (*_pack_sn_mr[4])(void);
    void (*_pack_sn_nr[4])(void);
    void (*_pack_ns_mr[4])(void);
    void (*_pack_ns_nr[4])(void);
    void (*_pack_ss_mr[4])(void);
    void (*_pack_ss_nr[4])(void);
    void (*_pack_nb_mr[4])(void);
    void (*_pack_nb_nr[4])(void);
    void (*_pack_sb_mr[4])(void);
    void (*_pack_sb_nr[4])(void);

    template <typename T>
    void pack_mr(len_type m, len_type k,
                 const T* p_a, stride_type rs_a, stride_type cs_a,
                 T* p_ap) const
    {
        pack_nn_ukr_t<T>(_pack_nn_mr[type_idx<T>::value])
            (m, k, p_a, rs_a, cs_a, p_ap);
    }

    template <typename T>
    void pack_nr(len_type m, len_type k,
                 const T* p_a, stride_type rs_a, stride_type cs_a,
                 T* p_ap) const
    {
        pack_nn_ukr_t<T>(_pack_nn_nr[type_idx<T>::value])
            (m, k, p_a, rs_a, cs_a, p_ap);
    }

    template <typename T>
    void pack_mr(len_type m, len_type k,
                 const T* p_a, const stride_type* rscat_a, stride_type cs_a,
                 T* p_ap) const
    {
        pack_sn_ukr_t<T>(_pack_sn_mr[type_idx<T>::value])
            (m, k, p_a, rscat_a, cs_a, p_ap);
    }

    template <typename T>
    void pack_nr(len_type m, len_type k,
                 const T* p_a, const stride_type* rscat_a, stride_type cs_a,
                 T* p_ap) const
    {
        pack_sn_ukr_t<T>(_pack_sn_nr[type_idx<T>::value])
            (m, k, p_a, rscat_a, cs_a, p_ap);
    }

    template <typename T>
    void pack_mr(len_type m, len_type k,
                 const T* p_a, stride_type rs_a, const stride_type* cscat_a,
                 T* p_ap) const
    {
        pack_ns_ukr_t<T>(_pack_ns_mr[type_idx<T>::value])
            (m, k, p_a, rs_a, cscat_a, p_ap);
    }

    template <typename T>
    void pack_nr(len_type m, len_type k,
                 const T* p_a, stride_type rs_a, const stride_type* cscat_a,
                 T* p_ap) const
    {
        pack_ns_ukr_t<T>(_pack_ns_nr[type_idx<T>::value])
            (m, k, p_a, rs_a, cscat_a, p_ap);
    }

    template <typename T>
    void pack_mr(len_type m, len_type k,
                 const T* p_a, const stride_type* rscat_a, const stride_type* cscat_a,
                 T* p_ap) const
    {
        pack_ss_ukr_t<T>(_pack_ss_mr[type_idx<T>::value])
            (m, k, p_a, rscat_a, cscat_a, p_ap);
    }

    template <typename T>
    void pack_nr(len_type m, len_type k,
                 const T* p_a, const stride_type* rscat_a, const stride_type* cscat_a,
                 T* p_ap) const
    {
        pack_ss_ukr_t<T>(_pack_ss_nr[type_idx<T>::value])
            (m, k, p_a, rscat_a, cscat_a, p_ap);
    }

    template <typename T>
    void pack_mr(len_type m, len_type k,
                 const T* p_a, stride_type rs_a, const stride_type* cscat_a,
                 const stride_type* cbs_a,
                 T* p_ap) const
    {
        pack_nb_ukr_t<T>(_pack_nb_mr[type_idx<T>::value])
            (m, k, p_a, rs_a, cscat_a, cbs_a, p_ap);
    }

    template <typename T>
    void pack_nr(len_type m, len_type k,
                 const T* p_a, stride_type rs_a, const stride_type* cscat_a,
                 const stride_type* cbs_a,
                 T* p_ap) const
    {
        pack_nb_ukr_t<T>(_pack_nb_nr[type_idx<T>::value])
            (m, k, p_a, rs_a, cscat_a, cbs_a, p_ap);
    }

    template <typename T>
    void pack_mr(len_type m, len_type k,
                 const T* p_a, const stride_type* rscat_a, const stride_type* cscat_a,
                 const stride_type* cbs_a,
                 T* p_ap) const
    {
        pack_sb_ukr_t<T>(_pack_sb_mr[type_idx<T>::value])
            (m, k, p_a, rscat_a, cscat_a, cbs_a, p_ap);
    }

    template <typename T>
    void pack_nr(len_type m, len_type k,
                 const T* p_a, const stride_type* rscat_a, const stride_type* cscat_a,
                 const stride_type* cbs_a,
                 T* p_ap) const
    {
        pack_sb_ukr_t<T>(_pack_sb_nr[type_idx<T>::value])
            (m, k, p_a, rscat_a, cscat_a, cbs_a, p_ap);
    }

    bool _row_major[4];

    template <typename T>
    bool row_major() const { return _row_major[type_idx<T>::value]; }

    check_fn_t check;

    template <typename Traits> config(const Traits&)
    : MR(typename Traits::MR()), NR(typename Traits::NR()), KR(typename Traits::KR()),
      MC(typename Traits::MC()), NC(typename Traits::NC()), KC(typename Traits::KC()),
      _gemm_ukr{Traits::gemm_ukr<   float>, Traits::gemm_ukr<  double>,
                Traits::gemm_ukr<scomplex>, Traits::gemm_ukr<dcomplex>},
      _pack_nn_mr{Traits::pack_nn_mr<   float>, Traits::pack_nn_mr<  double>,
                  Traits::pack_nn_mr<scomplex>, Traits::pack_nn_mr<dcomplex>},
      _pack_nn_nr{Traits::pack_nn_nr<   float>, Traits::pack_nn_nr<  double>,
                  Traits::pack_nn_nr<scomplex>, Traits::pack_nn_nr<dcomplex>},
      _pack_sn_mr{Traits::pack_sn_mr<   float>, Traits::pack_sn_mr<  double>,
                  Traits::pack_sn_mr<scomplex>, Traits::pack_sn_mr<dcomplex>},
      _pack_sn_nr{Traits::pack_sn_nr<   float>, Traits::pack_sn_nr<  double>,
                  Traits::pack_sn_nr<scomplex>, Traits::pack_sn_nr<dcomplex>},
      _pack_ns_mr{Traits::pack_ns_mr<   float>, Traits::pack_ns_mr<  double>,
                  Traits::pack_ns_mr<scomplex>, Traits::pack_ns_mr<dcomplex>},
      _pack_ns_nr{Traits::pack_ns_nr<   float>, Traits::pack_ns_nr<  double>,
                  Traits::pack_ns_nr<scomplex>, Traits::pack_ns_nr<dcomplex>},
      _pack_ss_mr{Traits::pack_ss_mr<   float>, Traits::pack_ss_mr<  double>,
                  Traits::pack_ss_mr<scomplex>, Traits::pack_ss_mr<dcomplex>},
      _pack_ss_nr{Traits::pack_ss_nr<   float>, Traits::pack_ss_nr<  double>,
                  Traits::pack_ss_nr<scomplex>, Traits::pack_ss_nr<dcomplex>},
      _pack_nb_mr{Traits::pack_nb_mr<   float>, Traits::pack_nb_mr<  double>,
                  Traits::pack_nb_mr<scomplex>, Traits::pack_nb_mr<dcomplex>},
      _pack_nb_nr{Traits::pack_nb_nr<   float>, Traits::pack_nb_nr<  double>,
                  Traits::pack_nb_nr<scomplex>, Traits::pack_nb_nr<dcomplex>},
      _pack_sb_mr{Traits::pack_sb_mr<   float>, Traits::pack_sb_mr<  double>,
                  Traits::pack_sb_mr<scomplex>, Traits::pack_sb_mr<dcomplex>},
      _pack_sb_nr{Traits::pack_sb_nr<   float>, Traits::pack_sb_nr<  double>,
                  Traits::pack_sb_nr<scomplex>, Traits::pack_sb_nr<dcomplex>},
      _row_major{Traits::template row_major<   float>::value, Traits::template row_major<   float>::value,
                 Traits::template row_major<scomplex>::value, Traits::template row_major<dcomplex>::value},
      check(Traits::check) {}
};

template <typename Config>
struct config_traits
{
    template <typename T> using _default_MR = simple_blocksize<T, 8, 4, 4, 2>;
    template <typename T> using _default_NR = simple_blocksize<T, 4, 4, 2, 2>;
    template <typename T> using _default_KR = simple_blocksize<T, 4, 2, 2, 1>;

    template <typename T> using MR = typename blocksize_traits<
        Config::template MR, _default_MR>::template type<T>;
    template <typename T> using NR = typename blocksize_traits<
        Config::template NR, _default_NR>::template type<T>;
    template <typename T> using KR = typename blocksize_traits<
        Config::template KR, _default_KR>::template type<T>;

    template <typename T> using _default_MC = simple_blocksize<T,  512,  256,  256,  128>;
    template <typename T> using _default_NC = simple_blocksize<T, 4096, 4096, 4096, 4096>;
    template <typename T> using _default_KC = simple_blocksize<T,  256,  256,  256,  256>;

    template <typename T> using MC = typename blocksize_traits<
        Config::template MC, _default_MC, MR>::template type<T>;
    template <typename T> using NC = typename blocksize_traits<
        Config::template NC, _default_NC, NR>::template type<T>;
    template <typename T> using KC = typename blocksize_traits<
        Config::template KC, _default_KC, KR>::template type<T>;

    template <typename T, int Dim> using BS =
        typename _BS<config_traits, Dim>::template type<T>;

    TBLIS_DEFAULT_VALUE_T(matrix_reduce_ker, matrix_reduce_ker_t<T>, Config, matrix_reduce_ker, matrix_reduce_ker_def<T>);

    TBLIS_DEFAULT_VALUE_T(gemm_ukr, gemm_ukr_t<T>, Config, gemm_ukr, (gemm_ukr_def<config_traits,T>));

    TBLIS_DEFAULT_VALUE_T(pack_nn_mr, pack_nn_ukr_t<T>, Config, pack_nn_mr, (pack_nn_ukr_def<config_traits,T,matrix_constants::MAT_A>));
    TBLIS_DEFAULT_VALUE_T(pack_nn_nr, pack_nn_ukr_t<T>, Config, pack_nn_nr, (pack_nn_ukr_def<config_traits,T,matrix_constants::MAT_B>));
    TBLIS_DEFAULT_VALUE_T(pack_sn_mr, pack_sn_ukr_t<T>, Config, pack_sn_mr, (pack_sn_ukr_def<config_traits,T,matrix_constants::MAT_A>));
    TBLIS_DEFAULT_VALUE_T(pack_sn_nr, pack_sn_ukr_t<T>, Config, pack_sn_nr, (pack_sn_ukr_def<config_traits,T,matrix_constants::MAT_B>));
    TBLIS_DEFAULT_VALUE_T(pack_ns_mr, pack_ns_ukr_t<T>, Config, pack_ns_mr, (pack_ns_ukr_def<config_traits,T,matrix_constants::MAT_A>));
    TBLIS_DEFAULT_VALUE_T(pack_ns_nr, pack_ns_ukr_t<T>, Config, pack_ns_nr, (pack_ns_ukr_def<config_traits,T,matrix_constants::MAT_B>));
    TBLIS_DEFAULT_VALUE_T(pack_ss_mr, pack_ss_ukr_t<T>, Config, pack_ss_mr, (pack_ss_ukr_def<config_traits,T,matrix_constants::MAT_A>));
    TBLIS_DEFAULT_VALUE_T(pack_ss_nr, pack_ss_ukr_t<T>, Config, pack_ss_nr, (pack_ss_ukr_def<config_traits,T,matrix_constants::MAT_B>));
    TBLIS_DEFAULT_VALUE_T(pack_nb_mr, pack_nb_ukr_t<T>, Config, pack_nb_mr, (pack_nb_ukr_def<config_traits,T,matrix_constants::MAT_A>));
    TBLIS_DEFAULT_VALUE_T(pack_nb_nr, pack_nb_ukr_t<T>, Config, pack_nb_nr, (pack_nb_ukr_def<config_traits,T,matrix_constants::MAT_B>));
    TBLIS_DEFAULT_VALUE_T(pack_sb_mr, pack_sb_ukr_t<T>, Config, pack_sb_mr, (pack_sb_ukr_def<config_traits,T,matrix_constants::MAT_A>));
    TBLIS_DEFAULT_VALUE_T(pack_sb_nr, pack_sb_ukr_t<T>, Config, pack_sb_nr, (pack_sb_ukr_def<config_traits,T,matrix_constants::MAT_B>));

    TBLIS_DEFAULT_VALUE_T(row_major, bool, Config, row_major, false);

    template <typename T> using pack_mr = pack_nn_mr<T>;
    template <typename T> using pack_nr = pack_nn_nr<T>;

    static check_fn_t check;
};

#define TBLIS_CONFIG(config) \
struct config##_ \
{ \
    template <typename T> struct MC {}; \
    template <typename T> struct NC {}; \
    template <typename T> struct KC {}; \
    template <typename T> struct MR {}; \
    template <typename T> struct NR {}; \
    template <typename T> struct KR {}; \
    \
    template <typename T> struct matrix_reduce_ker {}; \
    \
    template <typename T> struct gemm_ukr {}; \
    \
    template <typename T> struct pack_nn_mr {}; \
    template <typename T> struct pack_nn_nr {}; \
    template <typename T> struct pack_sn_mr {}; \
    template <typename T> struct pack_sn_nr {}; \
    template <typename T> struct pack_ns_mr {}; \
    template <typename T> struct pack_ns_nr {}; \
    template <typename T> struct pack_ss_mr {}; \
    template <typename T> struct pack_ss_nr {}; \
    template <typename T> struct pack_nb_mr {}; \
    template <typename T> struct pack_nb_nr {}; \
    template <typename T> struct pack_sb_mr {}; \
    template <typename T> struct pack_sb_nr {}; \
    \
    template <typename T> struct row_major {}; \
    static check_fn_t check; \
}; \
using config = config_traits<config##_>;

#define TBLIS_CONFIG_X_1(config, X, T, type1, name1, value1) \
template <> struct config::X<T> { static constexpr type1 name1 = value1; };
#define TBLIS_CONFIG_X_2(config, X, T, type1, name1, value1, \
                                       type2, name2, value2) \
template <> struct config::X<T> { static constexpr type1 name1 = value1; \
                                  static constexpr type2 name2 = value2; };

#define TBLIS_CONFIG_MC(config, T, _def) TBLIS_CONFIG_X_1(config, MC, T, len_type, def, _def)
#define TBLIS_CONFIG_NC(config, T, _def) TBLIS_CONFIG_X_1(config, NC, T, len_type, def, _def)
#define TBLIS_CONFIG_KC(config, T, _def) TBLIS_CONFIG_X_1(config, KC, T, len_type, def, _def)
#define TBLIS_CONFIG_MC_EX(config, T, _def, _max) TBLIS_CONFIG_X_2(config, MC, T, len_type, def, _def, len_type, max, _max)
#define TBLIS_CONFIG_NC_EX(config, T, _def, _max) TBLIS_CONFIG_X_2(config, NC, T, len_type, def, _def, len_type, max, _max)
#define TBLIS_CONFIG_KC_EX(config, T, _def, _max) TBLIS_CONFIG_X_2(config, KC, T, len_type, def, _def, len_type, max, _max)
#define TBLIS_CONFIG_MR(config, T, _def) TBLIS_CONFIG_X_1(config, MR, T, len_type, def, _def)
#define TBLIS_CONFIG_NR(config, T, _def) TBLIS_CONFIG_X_1(config, NR, T, len_type, def, _def)
#define TBLIS_CONFIG_KR(config, T, _def) TBLIS_CONFIG_X_1(config, KR, T, len_type, def, _def)
#define TBLIS_CONFIG_MR_EX(config, T, _def, _extent) TBLIS_CONFIG_X_2(config, MC, T, len_type, def, _def, len_type, extent, _extent)
#define TBLIS_CONFIG_NR_EX(config, T, _def, _extent) TBLIS_CONFIG_X_2(config, NC, T, len_type, def, _def, len_type, extent, _extent)
#define TBLIS_CONFIG_KR_EX(config, T, _def, _extent) TBLIS_CONFIG_X_2(config, KC, T, len_type, def, _def, len_type, extent, _extent)
#define TBLIS_CONFIG_GEMM_UKR(config, T, _gemm_ukr) TBLIS_CONFIG_X_1(config, gemm_ukr, T, gemm_ukr_t<T>, value, _gemm_ukr)
#define TBLIS_CONFIG_PACK_MR(config, T, variant, _pack_mr) TBLIS_CONFIG_X_1(config, pack_##variant##_mr, T, pack_##variant##_ukr_t<T>, value, _pack_mr)
#define TBLIS_CONFIG_PACK_NR(config, T, variant, _pack_nr) TBLIS_CONFIG_X_1(config, pack_##variant##_nr, T, pack_##variant##_ukr_t<T>, value, _pack_nr)
#define TBLIS_CONFIG_ROW_MAJOR(config, T) TBLIS_CONFIG_X_1(config, row_major, T, bool, value, true)
#define TBLIS_CONFIG_CHECK(config, chk) template <> check_fn_t config::check = chk;

const config* get_default_config();

}


#include "configs/dunnington/config.hpp"
#include "configs/sandybridge/config.hpp"
#include "configs/haswell/config.hpp"
#include "configs/knl/config.hpp"
#include "configs/bulldozer/config.hpp"
#include "configs/piledriver/config.hpp"
#include "configs/carrizo/config.hpp"
#include "configs/reference/config.hpp"

#endif
