#ifndef _TBLIS_CONFIGS_HPP_
#define _TBLIS_CONFIGS_HPP_

#include "util/tblis_basic_types.hpp"

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

template <typename T>
using gemm_ukr_t =
void (*)(stride_type k,
         const T* TBLIS_RESTRICT alpha,
         const T* TBLIS_RESTRICT a, const T* TBLIS_RESTRICT b,
         const T* TBLIS_RESTRICT beta,
         T* TBLIS_RESTRICT c, stride_type rs_c, stride_type cs_c);

template <typename Config, typename T>
void GenericMicroKernel(stride_type k,
                        const T* TBLIS_RESTRICT alpha,
                        const T* TBLIS_RESTRICT p_a, const T* TBLIS_RESTRICT p_b,
                        const T* TBLIS_RESTRICT beta,
                        T* TBLIS_RESTRICT p_c, stride_type rs_c, stride_type cs_c)
{
    constexpr len_type MR = Config::template MR<T>::def;
    constexpr len_type NR = Config::template NR<T>::def;

    T p_ab[MR*NR] __attribute__((aligned(64))) = {};

    while (k --> 0)
    {
        for (int j = 0;j < NR;j++)
        {
            for (int i = 0;i < MR;i++)
            {
                p_ab[i + MR*j] += p_a[i] * p_b[j];
            }
        }

        p_a += MR;
        p_b += NR;
    }

    if (*beta == T(0))
    {
        for (len_type j = 0;j < NR;j++)
        {
            for (len_type i = 0;i < MR;i++)
            {
                p_c[i*rs_c + j*cs_c] = (*alpha)*p_ab[i + MR*j];
            }
        }
    }
    else
    {
        for (len_type j = 0;j < NR;j++)
        {
            for (len_type i = 0;i < MR;i++)
            {
                p_c[i*rs_c + j*cs_c] = (*alpha)*p_ab[i + MR*j] +
                                       (*beta)*p_c[i*rs_c + j*cs_c];
            }
        }
    }
}

//TODO: threading over k

template <typename T>
using pack_nn_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, stride_type rs_a, stride_type cs_a,
         T* p_ap);

template <typename T>
using pack_sn_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, const stride_type* rscat_a, stride_type cs_a,
         T* p_ap);

template <typename T>
using pack_ns_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, stride_type rs_a, const stride_type* cscat_a,
         T* p_ap);

template <typename T>
using pack_ss_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, const stride_type* rscat_a, const stride_type* cscat_a,
         T* p_ap);

template <typename T>
using pack_nb_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, stride_type rs_a, const stride_type* cscat_a,
         const stride_type* cbs_a,
         T* p_ap);

template <typename T>
using pack_sb_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, const stride_type* rscat_a, const stride_type* cscat_a,
         const stride_type* cbs_a,
         T* p_ap);

template <typename T>
using pack_ukr_t = pack_nn_ukr_t<T>;

template <typename Config, typename T, int Mat>
void PackMicroPanel(len_type m, len_type k,
                    const T* TBLIS_RESTRICT p_a, stride_type rs_a, stride_type cs_a,
                    T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template MR<T>::def
                                          : Config::template NR<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template MR<T>::extent
                                          : Config::template NR<T>::extent);
    constexpr len_type KR = Config::template KR<T>::def;

    if (m == MR && rs_a == 1)
    {
        len_type p = 0;
        for (;p < k-KR;p += KR)
        {
            for (len_type kr = 0;kr < KR;kr++)
            {
                for (len_type mr = 0;mr < MR;mr++)
                {
                    p_ap[mr + ME*kr] = p_a[mr + cs_a*kr];
                }
            }

            p_a += cs_a*KR;
            p_ap += ME*KR;
        }

        for (len_type kr = 0;kr < k-p;kr++)
        {
            for (len_type mr = 0;mr < MR;mr++)
            {
                p_ap[mr + ME*kr] = p_a[mr + cs_a*kr];
            }
        }
    }
    else if (m == MR && cs_a == 1)
    {
        len_type p = 0;
        for (;p < k-KR;p += KR)
        {
            for (len_type kr = 0;kr < KR;kr++)
            {
                for (len_type mr = 0;mr < MR;mr++)
                {
                    p_ap[mr + ME*kr] = p_a[rs_a*mr + kr];
                }
            }

            p_a += KR;
            p_ap += ME*KR;
        }

        for (len_type kr = 0;kr < k-p;kr++)
        {
            for (len_type mr = 0;mr < MR;mr++)
            {
                p_ap[mr + ME*kr] = p_a[rs_a*mr + kr];
            }
        }
    }
    else
    {
        for (len_type p = 0;p < k;p++)
        {
            for (len_type mr = 0;mr < m;mr++)
            {
                p_ap[mr + ME*p] = p_a[rs_a*mr + cs_a*p];
            }

            for (len_type mr = m;mr < MR;mr++)
            {
                p_ap[mr + ME*p] = T();
            }
        }
    }
}

template <typename Config, typename T, int Mat>
void PackMicroPanel(len_type m, len_type k,
                    const T* TBLIS_RESTRICT p_a,
                    const stride_type* TBLIS_RESTRICT rscat_a, stride_type cs_a,
                    T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template MR<T>::def
                                          : Config::template NR<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template MR<T>::extent
                                          : Config::template NR<T>::extent);
    constexpr len_type KR = Config::template KR<T>::def;

    for (len_type p = 0;p < k;p++)
    {
        for (len_type mr = 0;mr < m;mr++)
        {
            p_ap[mr + ME*p] = p_a[rscat_a[mr] + cs_a*p];
        }

        for (len_type mr = m;mr < MR;mr++)
        {
            p_ap[mr + ME*p] = T();
        }
    }
}

template <typename Config, typename T, int Mat>
void PackMicroPanel(len_type m, len_type k,
                    const T* TBLIS_RESTRICT p_a,
                    stride_type rs_a, const stride_type* TBLIS_RESTRICT cscat_a,
                    T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template MR<T>::def
                                          : Config::template NR<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template MR<T>::extent
                                          : Config::template NR<T>::extent);
    constexpr len_type KR = Config::template KR<T>::def;

    for (len_type p = 0;p < k;p++)
    {
        for (len_type mr = 0;mr < m;mr++)
        {
            p_ap[mr + ME*p] = p_a[rs_a*mr + cscat_a[p]];
        }

        for (len_type mr = m;mr < MR;mr++)
        {
            p_ap[mr + ME*p] = T();
        }
    }
}

template <typename Config, typename T, int Mat>
void PackMicroPanel(len_type m, len_type k,
                    const T* TBLIS_RESTRICT p_a,
                    const stride_type* TBLIS_RESTRICT rscat_a,
                    const stride_type* TBLIS_RESTRICT cscat_a,
                    T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template MR<T>::def
                                          : Config::template NR<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template MR<T>::extent
                                          : Config::template NR<T>::extent);
    constexpr len_type KR = Config::template KR<T>::def;

    for (len_type p = 0;p < k;p++)
    {
        for (len_type mr = 0;mr < m;mr++)
        {
            p_ap[mr + ME*p] = p_a[rscat_a[mr] + cscat_a[p]];
        }

        for (len_type mr = m;mr < MR;mr++)
        {
            p_ap[mr + ME*p] = T();
        }
    }
}

template <typename Config, typename T, int Mat>
void PackMicroPanel(len_type m, len_type k,
                    const T* TBLIS_RESTRICT p_a,
                    stride_type rs_a, const stride_type* TBLIS_RESTRICT cscat_a,
                    const stride_type* TBLIS_RESTRICT cbs_a,
                    T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template MR<T>::def
                                          : Config::template NR<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template MR<T>::extent
                                          : Config::template NR<T>::extent);
    constexpr len_type KR = Config::template KR<T>::def;

    //TODO use block stride
    for (len_type p = 0;p < k;p++)
    {
        for (len_type mr = 0;mr < m;mr++)
        {
            p_ap[mr + ME*p] = p_a[rs_a*mr + cscat_a[p]];
        }

        for (len_type mr = m;mr < MR;mr++)
        {
            p_ap[mr + ME*p] = T();
        }
    }
}

template <typename Config, typename T, int Mat>
void PackMicroPanel(len_type m, len_type k,
                    const T* TBLIS_RESTRICT p_a,
                    const stride_type* TBLIS_RESTRICT rscat_a,
                    const stride_type* TBLIS_RESTRICT cscat_a,
                    const stride_type* TBLIS_RESTRICT cbs_a,
                    T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template MR<T>::def
                                          : Config::template NR<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template MR<T>::extent
                                          : Config::template NR<T>::extent);
    constexpr len_type KR = Config::template KR<T>::def;

    for (len_type p = 0;p < k;p++)
    {
        for (len_type mr = 0;mr < m;mr++)
        {
            p_ap[mr + ME*p] = p_a[rscat_a[mr] + cscat_a[p]];
        }

        for (len_type mr = m;mr < MR;mr++)
        {
            p_ap[mr + ME*p] = T();
        }
    }
}

#define TBLIS_CONFIG(config) \
struct config##_ \
{ \
    template <typename T> struct MC {}; \
    template <typename T> struct NC {}; \
    template <typename T> struct KC {}; \
    template <typename T> struct MR {}; \
    template <typename T> struct NR {}; \
    template <typename T> struct KR {}; \
    template <typename T> struct gemm_ukr {}; \
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

    template <typename T> using _default_MC = simple_blocksize<T,  512,  256,  256,  256>;
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

    TBLIS_DEFAULT_VALUE_T(  gemm_ukr,    gemm_ukr_t<T>, Config,   gemm_ukr, (                    GenericMicroKernel<config_traits,T>));
    TBLIS_DEFAULT_VALUE_T(pack_nn_mr, pack_nn_ukr_t<T>, Config, pack_nn_mr, (PackMicroPanel<config_traits,T,matrix_constants::MAT_A>));
    TBLIS_DEFAULT_VALUE_T(pack_nn_nr, pack_nn_ukr_t<T>, Config, pack_nn_nr, (PackMicroPanel<config_traits,T,matrix_constants::MAT_B>));
    TBLIS_DEFAULT_VALUE_T(pack_sn_mr, pack_sn_ukr_t<T>, Config, pack_sn_mr, (PackMicroPanel<config_traits,T,matrix_constants::MAT_A>));
    TBLIS_DEFAULT_VALUE_T(pack_sn_nr, pack_sn_ukr_t<T>, Config, pack_sn_nr, (PackMicroPanel<config_traits,T,matrix_constants::MAT_B>));
    TBLIS_DEFAULT_VALUE_T(pack_ns_mr, pack_ns_ukr_t<T>, Config, pack_ns_mr, (PackMicroPanel<config_traits,T,matrix_constants::MAT_A>));
    TBLIS_DEFAULT_VALUE_T(pack_ns_nr, pack_ns_ukr_t<T>, Config, pack_ns_nr, (PackMicroPanel<config_traits,T,matrix_constants::MAT_B>));
    TBLIS_DEFAULT_VALUE_T(pack_ss_mr, pack_ss_ukr_t<T>, Config, pack_ss_mr, (PackMicroPanel<config_traits,T,matrix_constants::MAT_A>));
    TBLIS_DEFAULT_VALUE_T(pack_ss_nr, pack_ss_ukr_t<T>, Config, pack_ss_nr, (PackMicroPanel<config_traits,T,matrix_constants::MAT_B>));
    TBLIS_DEFAULT_VALUE_T(pack_nb_mr, pack_nb_ukr_t<T>, Config, pack_nb_mr, (PackMicroPanel<config_traits,T,matrix_constants::MAT_A>));
    TBLIS_DEFAULT_VALUE_T(pack_nb_nr, pack_nb_ukr_t<T>, Config, pack_nb_nr, (PackMicroPanel<config_traits,T,matrix_constants::MAT_B>));
    TBLIS_DEFAULT_VALUE_T(pack_sb_mr, pack_sb_ukr_t<T>, Config, pack_sb_mr, (PackMicroPanel<config_traits,T,matrix_constants::MAT_A>));
    TBLIS_DEFAULT_VALUE_T(pack_sb_nr, pack_sb_ukr_t<T>, Config, pack_sb_nr, (PackMicroPanel<config_traits,T,matrix_constants::MAT_B>));
    TBLIS_DEFAULT_VALUE_T( row_major,             bool, Config,  row_major,                                                     false);

    template <typename T> using pack_mr = pack_nn_mr<T>;
    template <typename T> using pack_nr = pack_nn_nr<T>;

    static check_fn_t check;
};

}

#endif
