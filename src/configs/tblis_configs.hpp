#ifndef _TBLIS_CONFIGS_HPP_
#define _TBLIS_CONFIGS_HPP_

#include "tblis_basic_types.hpp"

namespace tblis
{

namespace matrix_constants
{
    enum {MAT_A, MAT_B, MAT_C};
    enum {DIM_MR=0x2, DIM_NR=0x4, DIM_KR=0x8,
          DIM_MC=0x3, DIM_NC=0x5, DIM_KC=0x9,
          DIM_M=DIM_MR, DIM_N=DIM_NR, DIM_K=DIM_KR};
}

template <typename T>
using gemm_ukr_t =
void (*)(stride_type k,
         const T* restrict alpha,
         const T* restrict a, const T* restrict b,
         const T* restrict beta,
         T* restrict c, stride_type rs_c, stride_type cs_c);

template <typename T, idx_type MR, idx_type NR>
void GenericMicroKernel(stride_type k,
                        const T* restrict alpha,
                        const T* restrict p_a, const T* restrict p_b,
                        const T* restrict beta,
                        T* restrict p_c, stride_type rs_c, stride_type cs_c);

template <typename T>
using pack_ukr_t =
void (*)(idx_type m, idx_type k,
         const T* restrict & p_a, stride_type rs_a, stride_type cs_a,
         T* restrict & p_ap);

template <typename T, idx_type MR, idx_type KR>
void PackMicroPanel(idx_type m, idx_type k,
                    const T* restrict & p_a, stride_type rs_a, stride_type cs_a,
                    T* restrict & p_ap);

#define TBLIS_CONFIG(config) \
struct config \
{ \
    template <typename T> struct MC {}; \
    template <typename T> struct NC {}; \
    template <typename T> struct KC {}; \
    template <typename T> struct MR {}; \
    template <typename T> struct NR {}; \
    template <typename T> struct KR {}; \
    template <typename T> struct gemm_ukr {}; \
    template <typename T> struct pack_mr {}; \
    template <typename T> struct pack_nr {}; \
    template <typename T> struct row_major {}; \
};

#define TBLIS_CONFIG_X_1(config, X, T, type1, name1, value1) \
template <> struct config::X<T> { static constexpr type1 name1 = value1; };
#define TBLIS_CONFIG_X_2(config, X, T, type1, name1, value1, \
                                       type2, name2, value2) \
template <> struct config::X<T> { static constexpr type1 name1 = value1; \
                                  static constexpr type2 name2 = value2; };

#define TBLIS_CONFIG_MC(config, T, _def) TBLIS_CONFIG_X_1(config, MC, T, idx_type, def, _def)
#define TBLIS_CONFIG_NC(config, T, _def) TBLIS_CONFIG_X_1(config, NC, T, idx_type, def, _def)
#define TBLIS_CONFIG_KC(config, T, _def) TBLIS_CONFIG_X_1(config, KC, T, idx_type, def, _def)
#define TBLIS_CONFIG_MC_EX(config, T, _def, _max) TBLIS_CONFIG_X_2(config, MC, T, idx_type, def, _def, idx_type, max, _max)
#define TBLIS_CONFIG_NC_EX(config, T, _def, _max) TBLIS_CONFIG_X_2(config, NC, T, idx_type, def, _def, idx_type, max, _max)
#define TBLIS_CONFIG_KC_EX(config, T, _def, _max) TBLIS_CONFIG_X_2(config, KC, T, idx_type, def, _def, idx_type, max, _max)
#define TBLIS_CONFIG_MR(config, T, _def) TBLIS_CONFIG_X_1(config, MR, T, idx_type, def, _def)
#define TBLIS_CONFIG_NR(config, T, _def) TBLIS_CONFIG_X_1(config, NR, T, idx_type, def, _def)
#define TBLIS_CONFIG_KR(config, T, _def) TBLIS_CONFIG_X_1(config, KR, T, idx_type, def, _def)
#define TBLIS_CONFIG_MR_EX(config, T, _def, _extent) TBLIS_CONFIG_X_2(config, MC, T, idx_type, def, _def, idx_type, extent, _extent)
#define TBLIS_CONFIG_NR_EX(config, T, _def, _extent) TBLIS_CONFIG_X_2(config, NC, T, idx_type, def, _def, idx_type, extent, _extent)
#define TBLIS_CONFIG_KR_EX(config, T, _def, _extent) TBLIS_CONFIG_X_2(config, KC, T, idx_type, def, _def, idx_type, extent, _extent)
#define TBLIS_CONFIG_GEMM_UKR(config, T, _gemm_ukr) TBLIS_CONFIG_X_1(config, gemm_ukr, T, gemm_ukr_t<T>, value, _gemm_ukr)
#define TBLIS_CONFIG_PACK_MR(config, T, _pack_mr) TBLIS_CONFIG_X_1(config, pack_mr, T, pack_ukr_t<T>, value, _pack_mr)
#define TBLIS_CONFIG_PACK_NR(config, T, _pack_nr) TBLIS_CONFIG_X_1(config, pack_nr, T, pack_ukr_t<T>, value, _pack_nr)
#define TBLIS_CONFIG_ROW_MAJOR(config, T, _row_major) TBLIS_CONFIG_X_1(config, row_major, T, bool, value, _row_major)

struct reference_config;

template <template <typename> class BS,
          template <typename> class BS_Ref,
          template <typename> class BS_Iota=BS>
struct blocksize_traits
{
    template <typename T>
    struct type
    {
        private:
            template<typename U>
            static std::integral_constant<idx_type,        U::def> _def_helper(U*);
            static std::integral_constant<idx_type,BS_Ref<T>::def> _def_helper(...);
        public:
            static constexpr idx_type def = decltype(_def_helper((BS<T>*)0))::value;

        private:
            template<typename U>
            static std::integral_constant<idx_type, U::max> _max_helper(U*);
            static std::integral_constant<idx_type,    def> _max_helper(...);
        public:
            static constexpr idx_type max = decltype(_max_helper((BS<T>*)0))::value;

        private:
            static constexpr idx_type _iota_def = decltype(_def_helper((BS_Iota<T>*)0))::value;
            template<typename U>
            static std::integral_constant<idx_type,   U::iota> _iota_helper(U*);
            static std::integral_constant<idx_type, _iota_def> _iota_helper(...);
        public:
            static constexpr idx_type iota = decltype(_iota_helper((BS<T>*)0))::value;

        private:
            template<typename U>
            static std::integral_constant<idx_type, U::extent> _extent_helper(U*);
            static std::integral_constant<idx_type,       def> _extent_helper(...);
        public:
            static constexpr idx_type extent = decltype(_extent_helper((BS<T>*)0))::value;
    };
};

template <typename Config>
struct config_traits
{
    template <typename T> using MR = typename blocksize_traits<
        Config::template MR, reference_config::MR>::template type<T>;
    template <typename T> using NR = typename blocksize_traits<
        Config::template NR, reference_config::NR>::template type<T>;
    template <typename T> using KR = typename blocksize_traits<
        Config::template KR, reference_config::KR>::template type<T>;

    template <typename T> using MC = typename blocksize_traits<
        Config::template MC, reference_config::MC, MR>::template type<T>;
    template <typename T> using NC = typename blocksize_traits<
        Config::template NC, reference_config::NC, NR>::template type<T>;
    template <typename T> using KC = typename blocksize_traits<
        Config::template KC, reference_config::KC, KR>::template type<T>;

    template <int Dim> struct _BS;
    template <typename T, int Dim> using BS = typename _BS<Dim>::template type<T>;

    private:
        template <typename U, typename T>
        static std::integral_constant<gemm_ukr_t<T>,                       U::gemm_ukr<T>::value> _gemm_ukr_helper(U*);
        template <typename U, typename T>
        static std::integral_constant<gemm_ukr_t<T>, GenericMicroKernel<T,MR<T>::def,NR<T>::def>> _gemm_ukr_helper(...);
    public:
        template <typename T>
        using gemm_ukr = std::integral_constant<gemm_ukr_t<T>,decltype(_gemm_ukr_helper<Config, T>((Config*)0))::value>;

    private:
        template <typename U, typename T>
        static std::integral_constant<pack_ukr_t<T>,                                   U::pack_mr<T>::value> _pack_mr_helper(U*);
        template <typename U, typename T>
        static std::integral_constant<pack_ukr_t<T>, (pack_ukr_t<T>)PackMicroPanel<T,MR<T>::def,KR<T>::def>> _pack_mr_helper(...);
    public:
        template <typename T>
        using pack_mr = std::integral_constant<pack_ukr_t<T>,decltype(_pack_mr_helper<Config, T>((Config*)0))::value>;

    private:
        template <typename U, typename T>
        static std::integral_constant<pack_ukr_t<T>,                                   U::pack_nr<T>::value> _pack_nr_helper(U*);
        template <typename U, typename T>
        static std::integral_constant<pack_ukr_t<T>, (pack_ukr_t<T>)PackMicroPanel<T,NR<T>::def,KR<T>::def>> _pack_nr_helper(...);
    public:
        template <typename T>
        using pack_nr = std::integral_constant<pack_ukr_t<T>,decltype(_pack_nr_helper<Config, T>((Config*)0))::value>;

    private:
        template <typename U, typename T>
        static std::integral_constant<bool, U::row_major<T>::value> _row_major_helper(U*);
        template <typename U, typename T>
        static std::integral_constant<bool,                  false> _row_major_helper(...);
    public:
        template <typename T>
        using row_major = std::integral_constant<bool,decltype(_row_major_helper<Config, T>((Config*)0))::value>;
};

template <typename Config>
template <>
struct config_traits<Config>::_BS<matrix_constants::DIM_MC>
{
        template <typename T> using type = MC<T>;
};

template <typename Config>
template <>
struct config_traits<Config>::_BS<matrix_constants::DIM_MR>
{
        template <typename T> using type = MR<T>;
};

template <typename Config>
template <>
struct config_traits<Config>::_BS<matrix_constants::DIM_NC>
{
        template <typename T> using type = NC<T>;
};

template <typename Config>
template <>
struct config_traits<Config>::_BS<matrix_constants::DIM_NR>
{
        template <typename T> using type = NR<T>;
};

template <typename Config>
template <>
struct config_traits<Config>::_BS<matrix_constants::DIM_KC>
{
        template <typename T> using type = KC<T>;
};

template <typename Config>
template <>
struct config_traits<Config>::_BS<matrix_constants::DIM_KR>
{
        template <typename T> using type = KR<T>;
};

}

#endif
