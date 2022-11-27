#ifndef TBLIS_INTERNAL_CONFIGS_HPP
#define TBLIS_INTERNAL_CONFIGS_HPP 1

#include <tblis/blis.h>

#include <tblis/base/configs.h>

#include <tblis/kernels/1v/add.hpp>
#include <tblis/kernels/1v/dot.hpp>
#include <tblis/kernels/1v/mult.hpp>
#include <tblis/kernels/1v/reduce.hpp>
#include <tblis/kernels/1v/scale.hpp>
#include <tblis/kernels/1v/set.hpp>
#include <tblis/kernels/1v/shift.hpp>

#include <tblis/kernels/1f/addf.hpp>
#include <tblis/kernels/1f/dotf.hpp>

#include <tblis/kernels/1m/trans.hpp>

#include <tblis/kernels/3m/gemm.hpp>
#include <tblis/kernels/3m/packm.hpp>

#define TBLIS_HAS_COMMA_HELPER(_0,_1,_2,_3,_4,_5,_6,...) _6
#define TBLIS_HAS_COMMA(...) TBLIS_HAS_COMMA_HELPER(__VA_ARGS__,1,1,1,1,1,0,barf)
#define TBLIS_COMMA_IF_EMPTY() ,,,,,
#define TBLIS_COMMA_IF_EMPTY_() ,,,,,
#define TBLIS_IS_EMPTY(x) TBLIS_HAS_COMMA(TBLIS_PASTE(TBLIS_COMMA_IF_EMPTY,x)())
#define TBLIS_GET_VALUE_IF_CASE_0(test,yes,no) yes
#define TBLIS_GET_VALUE_IF_CASE_1(test,yes,no) no
#define TBLIS_GET_VALUE_IF_CASE(test,yes,no,case) \
    TBLIS_PASTE(TBLIS_GET_VALUE_IF_CASE_,case)(test,yes,no)
#define TBLIS_GET_VALUE_IF(test,yes,no) \
    TBLIS_GET_VALUE_IF_CASE(test,yes,no,TBLIS_IS_EMPTY(test))
#define TBLIS_GET_VALUE_OR_DEFAULT(value,default) \
    TBLIS_GET_VALUE_IF(value,value,default)

#define TBLIS_BEGIN_CONFIG(cfg) \
struct cfg##_config : config_template<cfg##_config> \
{ \
    typedef cfg##_config this_config; \
 \
    static constexpr const char* name = #cfg; \
 \
    static const config& instance();

#define TBLIS_END_CONFIG };

#define TBLIS_CONFIG_INSTANTIATE(cfg) \
const config& cfg##_config::instance() \
{ \
    static config _instance(cfg##_config{}); \
    return _instance; \
}

#define TBLIS_CONFIG_REGISTER_BLOCKSIZE(name, S,D,C,Z, SE,DE,CE,ZE, SD,DD,CD,ZD, SED,DED,CED,ZED) \
    template <typename T> struct name : register_blocksize<T, \
        TBLIS_GET_VALUE_OR_DEFAULT(S,SD), \
        TBLIS_GET_VALUE_OR_DEFAULT(D,DD), \
        TBLIS_GET_VALUE_OR_DEFAULT(C,CD), \
        TBLIS_GET_VALUE_OR_DEFAULT(Z,ZD), \
        TBLIS_GET_VALUE_OR_DEFAULT(SE,TBLIS_GET_VALUE_OR_DEFAULT(S,SED)), \
        TBLIS_GET_VALUE_OR_DEFAULT(DE,TBLIS_GET_VALUE_OR_DEFAULT(D,DED)), \
        TBLIS_GET_VALUE_OR_DEFAULT(CE,TBLIS_GET_VALUE_OR_DEFAULT(C,CED)), \
        TBLIS_GET_VALUE_OR_DEFAULT(ZE,TBLIS_GET_VALUE_OR_DEFAULT(Z,ZED))> {};

#define TBLIS_CONFIG_CACHE_BLOCKSIZE(name, RB, S,D,C,Z, SM,DM,CM,ZM, SD,DD,CD,ZD, SED,DED,CED,ZED) \
    template <typename T> struct name : cache_blocksize<T, RB<T>, \
        TBLIS_GET_VALUE_OR_DEFAULT(S,SD), \
        TBLIS_GET_VALUE_OR_DEFAULT(D,DD), \
        TBLIS_GET_VALUE_OR_DEFAULT(C,CD), \
        TBLIS_GET_VALUE_OR_DEFAULT(Z,ZD), \
        TBLIS_GET_VALUE_OR_DEFAULT(SM,TBLIS_GET_VALUE_OR_DEFAULT(S,SED)), \
        TBLIS_GET_VALUE_OR_DEFAULT(DM,TBLIS_GET_VALUE_OR_DEFAULT(D,DED)), \
        TBLIS_GET_VALUE_OR_DEFAULT(CM,TBLIS_GET_VALUE_OR_DEFAULT(C,CED)), \
        TBLIS_GET_VALUE_OR_DEFAULT(ZM,TBLIS_GET_VALUE_OR_DEFAULT(Z,ZED))> {};

#define TBLIS_CONFIG_ADDF_NF(S,D,C,Z) \
    TBLIS_CONFIG_REGISTER_BLOCKSIZE(addf_nf, S,D,C,Z, S,D,C,Z, 4,4,4,4, 4,4,4,4)
#define TBLIS_CONFIG_DOTF_NF(S,D,C,Z) \
    TBLIS_CONFIG_REGISTER_BLOCKSIZE(dotf_nf, S,D,C,Z, S,D,C,Z, 4,4,4,4, 4,4,4,4)

#define TBLIS_CONFIG_TRANS_MR(S,D,C,Z) \
    TBLIS_CONFIG_REGISTER_BLOCKSIZE(trans_mr, S,D,C,Z, S,D,C,Z, 8,4,4,4, 8,4,4,4)
#define TBLIS_CONFIG_TRANS_NR(S,D,C,Z) \
    TBLIS_CONFIG_REGISTER_BLOCKSIZE(trans_nr, S,D,C,Z, S,D,C,Z, 4,4,4,2, 4,4,4,2)

#define TBLIS_CONFIG_GEMM_MR(S,D,C,Z) \
    TBLIS_CONFIG_REGISTER_BLOCKSIZE(gemm_mr, S,D,C,Z, S,D,C,Z, 8,4,4,2, 8,4,4,2)
#define TBLIS_CONFIG_GEMM_NR(S,D,C,Z) \
    TBLIS_CONFIG_REGISTER_BLOCKSIZE(gemm_nr, S,D,C,Z, S,D,C,Z, 4,4,2,2, 4,4,2,2)
#define TBLIS_CONFIG_GEMM_KR(S,D,C,Z) \
    TBLIS_CONFIG_REGISTER_BLOCKSIZE(gemm_kr, S,D,C,Z, 1,1,1,1, 4,2,2,1, 1,1,1,1)

#define TBLIS_CONFIG_GEMM_MR_EXTENT(S,D,C,Z, SE,DE,CE,ZE) \
    TBLIS_CONFIG_REGISTER_BLOCKSIZE(gemm_mr, S,D,C,Z, SE,DE,CE,ZE, 8,4,4,2, 8,4,4,2)
#define TBLIS_CONFIG_GEMM_NR_EXTENT(S,D,C,Z, SE,DE,CE,ZE) \
    TBLIS_CONFIG_REGISTER_BLOCKSIZE(gemm_nr, S,D,C,Z, SE,DE,CE,ZE, 4,4,2,2, 4,4,2,2)
#define TBLIS_CONFIG_GEMM_KR_EXTENT(S,D,C,Z, SE,DE,CE,ZE) \
    TBLIS_CONFIG_REGISTER_BLOCKSIZE(gemm_kr, S,D,C,Z, SE,DE,CE,ZE, 4,2,2,1, 1,1,1,1)

#define TBLIS_CONFIG_GEMM_MC(S,D,C,Z) \
    TBLIS_CONFIG_CACHE_BLOCKSIZE(gemm_mc, gemm_mr, S,D,C,Z, S,D,C,Z,  512,  256,  256,  128,  512,  256,  256,  128)
#define TBLIS_CONFIG_GEMM_NC(S,D,C,Z) \
    TBLIS_CONFIG_CACHE_BLOCKSIZE(gemm_nc, gemm_nr, S,D,C,Z, S,D,C,Z, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)
#define TBLIS_CONFIG_GEMM_KC(S,D,C,Z) \
    TBLIS_CONFIG_CACHE_BLOCKSIZE(gemm_kc, gemm_kr, S,D,C,Z, S,D,C,Z,  256,  256,  256,  256,  256,  256,  256,  256)

#define TBLIS_CONFIG_GEMM_MC_MAX(S,D,C,Z, SE,DE,CE,ZE) \
    TBLIS_CONFIG_CACHE_BLOCKSIZE(gemm_mc, gemm_mr, S,D,C,Z, SE,DE,CE,ZE,  512,  256,  256,  128,  512,  256,  256,  128)
#define TBLIS_CONFIG_GEMM_NC_MAX(S,D,C,Z, SE,DE,CE,ZE) \
    TBLIS_CONFIG_CACHE_BLOCKSIZE(gemm_nc, gemm_nr, S,D,C,Z, SE,DE,CE,ZE, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)
#define TBLIS_CONFIG_GEMM_KC_MAX(S,D,C,Z, SE,DE,CE,ZE) \
    TBLIS_CONFIG_CACHE_BLOCKSIZE(gemm_kc, gemm_kr, S,D,C,Z, SE,DE,CE,ZE,  256,  256,  256,  256,  256,  256,  256,  256)

#define TBLIS_CONFIG_PARAMETER(name, type, S,D,C,Z, SD,DD,CD,ZD) \
    template <typename T> struct name : static_value<T, type, \
        TBLIS_GET_VALUE_OR_DEFAULT(S,SD), \
        TBLIS_GET_VALUE_OR_DEFAULT(D,DD), \
        TBLIS_GET_VALUE_OR_DEFAULT(C,CD), \
        TBLIS_GET_VALUE_OR_DEFAULT(Z,ZD)> {};

#define TBLIS_CONFIG_GEMM_ROW_MAJOR(S,D,C,Z) \
    TBLIS_CONFIG_PARAMETER(gemm_row_major, bool, S,D,C,Z, false,false,false,false)
#define TBLIS_CONFIG_GEMM_FLIP_UKR(S,D,C,Z) \
    TBLIS_CONFIG_PARAMETER(gemm_flip_ukr, bool, S,D,C,Z, false,false,false,false)

#define TBLIS_CONFIG_M_THREAD_RATIO(S,D,C,Z) \
    TBLIS_CONFIG_PARAMETER(m_thread_ratio, int, S,D,C,Z, 2,2,2,2)
#define TBLIS_CONFIG_N_THREAD_RATIO(S,D,C,Z) \
    TBLIS_CONFIG_PARAMETER(n_thread_ratio, int, S,D,C,Z, 1,1,1,1)
#define TBLIS_CONFIG_MR_MAX_THREAD(S,D,C,Z) \
    TBLIS_CONFIG_PARAMETER(mr_max_thread, int, S,D,C,Z, 1,1,1,1)
#define TBLIS_CONFIG_NR_MAX_THREAD(S,D,C,Z) \
    TBLIS_CONFIG_PARAMETER(nr_max_thread, int, S,D,C,Z, 3,3,3,3)

#define TBLIS_CONFIG_UKR(name, type, S,D,C,Z, def_ker) \
    template <typename T> struct name : static_microkernel<T, type, \
        TBLIS_GET_VALUE_OR_DEFAULT(S,def_ker<   float>), \
        TBLIS_GET_VALUE_OR_DEFAULT(D,def_ker<  double>), \
        TBLIS_GET_VALUE_OR_DEFAULT(C,def_ker<scomplex>), \
        TBLIS_GET_VALUE_OR_DEFAULT(Z,def_ker<dcomplex>)> {};

#define TBLIS_CONFIG_UKR2(config, name, type, S,D,C,Z, def_ker) \
    template <typename T> struct name : static_microkernel<T, type, \
        TBLIS_GET_VALUE_OR_DEFAULT(S,(def_ker<config,   float>)), \
        TBLIS_GET_VALUE_OR_DEFAULT(D,(def_ker<config,  double>)), \
        TBLIS_GET_VALUE_OR_DEFAULT(C,(def_ker<config,scomplex>)), \
        TBLIS_GET_VALUE_OR_DEFAULT(Z,(def_ker<config,dcomplex>))> {};

#define TBLIS_CONFIG_UKR3(config, mat, name, type, S,D,C,Z, def_ker) \
    template <typename T> struct name : static_microkernel<T, type, \
        TBLIS_GET_VALUE_OR_DEFAULT(S,(def_ker<config,   float,mat>)), \
        TBLIS_GET_VALUE_OR_DEFAULT(D,(def_ker<config,  double,mat>)), \
        TBLIS_GET_VALUE_OR_DEFAULT(C,(def_ker<config,scomplex,mat>)), \
        TBLIS_GET_VALUE_OR_DEFAULT(Z,(def_ker<config,dcomplex,mat>))> {};

#define TBLIS_CONFIG_TRANS_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR2(this_config, trans_ukr, trans_ukr_t, S,D,C,Z, trans_ukr_def)

#define TBLIS_CONFIG_ADD_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR2(this_config, add_ukr, add_ukr_t, S,D,C,Z, add_ukr_def)
#define TBLIS_CONFIG_DOT_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR2(this_config, dot_ukr, dot_ukr_t, S,D,C,Z, dot_ukr_def)
#define TBLIS_CONFIG_MULT_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR2(this_config, mult_ukr, mult_ukr_t, S,D,C,Z, mult_ukr_def)
#define TBLIS_CONFIG_REDUCE_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR2(this_config, reduce_ukr, reduce_ukr_t, S,D,C,Z, reduce_ukr_def)
#define TBLIS_CONFIG_SCALE_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR2(this_config, scale_ukr, scale_ukr_t, S,D,C,Z, scale_ukr_def)
#define TBLIS_CONFIG_SET_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR2(this_config, set_ukr, set_ukr_t, S,D,C,Z, set_ukr_def)
#define TBLIS_CONFIG_SHIFT_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR2(this_config, shift_ukr, shift_ukr_t, S,D,C,Z, shift_ukr_def)

#define TBLIS_CONFIG_ADDF_SUM_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR2(this_config, addf_sum_ukr, addf_sum_ukr_t, S,D,C,Z, addf_sum_ukr_def)
#define TBLIS_CONFIG_ADDF_REP_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR2(this_config, addf_rep_ukr, addf_rep_ukr_t, S,D,C,Z, addf_rep_ukr_def)
#define TBLIS_CONFIG_DOTF_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR2(this_config, dotf_ukr, dotf_ukr_t, S,D,C,Z, dotf_ukr_def)

#define TBLIS_CONFIG_GEMM_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR2(this_config, gemm_ukr, gemm_ukr_t, S,D,C,Z, gemm_ukr_def)

#define TBLIS_CONFIG_GEMM_WRAP_UKR(S,D,C,Z) \
    template <typename T> struct gemm_ukr : static_microkernel<T, gemm_ukr_t, \
        TBLIS_GET_VALUE_IF(S,(blis_wrap_gemm_ukr<this_config,   float,S>),(gemm_ukr_def<this_config,   float>)), \
        TBLIS_GET_VALUE_IF(D,(blis_wrap_gemm_ukr<this_config,  double,D>),(gemm_ukr_def<this_config,  double>)), \
        TBLIS_GET_VALUE_IF(C,(blis_wrap_gemm_ukr<this_config,scomplex,C>),(gemm_ukr_def<this_config,scomplex>)), \
        TBLIS_GET_VALUE_IF(Z,(blis_wrap_gemm_ukr<this_config,dcomplex,Z>),(gemm_ukr_def<this_config,dcomplex>))> {};

#define TBLIS_CONFIG_PACK_NN_MR_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR3(this_config, matrix_constants::MAT_A, pack_nn_mr_ukr, pack_nn_ukr_t, S,D,C,Z, pack_nn_ukr_def)
#define TBLIS_CONFIG_PACK_NN_NR_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR3(this_config, matrix_constants::MAT_B, pack_nn_nr_ukr, pack_nn_ukr_t, S,D,C,Z, pack_nn_ukr_def)
#define TBLIS_CONFIG_PACK_SS_MR_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR3(this_config, matrix_constants::MAT_A, pack_ss_mr_ukr, pack_ss_ukr_t, S,D,C,Z, pack_ss_ukr_def)
#define TBLIS_CONFIG_PACK_SS_NR_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR3(this_config, matrix_constants::MAT_B, pack_ss_nr_ukr, pack_ss_ukr_t, S,D,C,Z, pack_ss_ukr_def)
#define TBLIS_CONFIG_PACK_NB_MR_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR3(this_config, matrix_constants::MAT_A, pack_nb_mr_ukr, pack_nb_ukr_t, S,D,C,Z, pack_nb_ukr_def)
#define TBLIS_CONFIG_PACK_NB_NR_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR3(this_config, matrix_constants::MAT_B, pack_nb_nr_ukr, pack_nb_ukr_t, S,D,C,Z, pack_nb_ukr_def)
#define TBLIS_CONFIG_PACK_SS_SCAL_MR_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR3(this_config, matrix_constants::MAT_A, pack_ss_scal_mr_ukr, pack_ss_scal_ukr_t, S,D,C,Z, pack_ss_scal_ukr_def)
#define TBLIS_CONFIG_PACK_SS_SCAL_NR_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR3(this_config, matrix_constants::MAT_B, pack_ss_scal_nr_ukr, pack_ss_scal_ukr_t, S,D,C,Z, pack_ss_scal_ukr_def)

#define TBLIS_CONFIG_UPDATE_NN_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR2(this_config, update_nn_ukr, update_nn_ukr_t, S,D,C,Z, update_nn_ukr_def)
#define TBLIS_CONFIG_UPDATE_SS_UKR(S,D,C,Z) \
    TBLIS_CONFIG_UKR2(this_config, update_ss_ukr, update_ss_ukr_t, S,D,C,Z, update_ss_ukr_def)

#define TBLIS_CONFIG_CHECK(func) static constexpr check_fn_t check = func;

namespace tblis
{

template <typename T, typename U, U S, U D, U C, U Z>
struct static_value
{
    static constexpr U value = std::is_same<T,   float>::value ? S :
                               std::is_same<T,  double>::value ? D :
                               std::is_same<T,scomplex>::value ? C :
                                                                 Z;
};

template <typename T, len_type S, len_type D, len_type C, len_type Z,
          len_type SE=S, len_type DE=D, len_type CE=C, len_type ZE=Z>
struct register_blocksize
{
    static constexpr len_type def = std::is_same<T,   float>::value ? S :
                                    std::is_same<T,  double>::value ? D :
                                    std::is_same<T,scomplex>::value ? C :
                                                                      Z;
    static constexpr len_type extent = std::is_same<T,   float>::value ? SE :
                                       std::is_same<T,  double>::value ? DE :
                                       std::is_same<T,scomplex>::value ? CE :
                                                                         ZE;
    static constexpr len_type iota = def;
    static constexpr len_type max = def;
};

template <typename T, typename RB, len_type S, len_type D, len_type C, len_type Z,
          len_type SM=S, len_type DM=D, len_type CM=C, len_type ZM=Z>
struct cache_blocksize
{
    static constexpr len_type def = std::is_same<T,   float>::value ? S :
                                    std::is_same<T,  double>::value ? D :
                                    std::is_same<T,scomplex>::value ? C :
                                                                      Z;
    static constexpr len_type extent = def;
    static constexpr len_type iota = RB::def;
    static constexpr len_type max = std::is_same<T,   float>::value ? SM :
                                    std::is_same<T,  double>::value ? DM :
                                    std::is_same<T,scomplex>::value ? CM :
                                                                      ZM;
};

template <typename T, typename kernel, kernel S, kernel D, kernel C, kernel Z>
struct static_microkernel;

template <typename kernel, kernel S, kernel D, kernel C, kernel Z>
struct static_microkernel<float, kernel, S, D, C, Z>
{
    static constexpr kernel value = S;
};

template <typename kernel, kernel S, kernel D, kernel C, kernel Z>
struct static_microkernel<double, kernel, S, D, C, Z>
{
    static constexpr kernel value = D;
};

template <typename kernel, kernel S, kernel D, kernel C, kernel Z>
struct static_microkernel<scomplex, kernel, S, D, C, Z>
{
    static constexpr kernel value = C;
};

template <typename kernel, kernel S, kernel D, kernel C, kernel Z>
struct static_microkernel<dcomplex, kernel, S, D, C, Z>
{
    static constexpr kernel value = Z;
};

template <typename Config>
struct config_template
{
    typedef Config this_config;

    TBLIS_CONFIG_ADD_UKR(_,_,_,_)
    TBLIS_CONFIG_DOT_UKR(_,_,_,_)
    TBLIS_CONFIG_MULT_UKR(_,_,_,_)
    TBLIS_CONFIG_REDUCE_UKR(_,_,_,_)
    TBLIS_CONFIG_SCALE_UKR(_,_,_,_)
    TBLIS_CONFIG_SET_UKR(_,_,_,_)
    TBLIS_CONFIG_SHIFT_UKR(_,_,_,_)

    TBLIS_CONFIG_ADDF_NF(_,_,_,_)
    TBLIS_CONFIG_DOTF_NF(_,_,_,_)
    TBLIS_CONFIG_ADDF_SUM_UKR(_,_,_,_)
    TBLIS_CONFIG_ADDF_REP_UKR(_,_,_,_)
    TBLIS_CONFIG_DOTF_UKR(_,_,_,_)

    TBLIS_CONFIG_TRANS_MR(_,_,_,_)
    TBLIS_CONFIG_TRANS_NR(_,_,_,_)
    TBLIS_CONFIG_TRANS_UKR(_,_,_,_)

    TBLIS_CONFIG_GEMM_MR(_,_,_,_)
    TBLIS_CONFIG_GEMM_NR(_,_,_,_)
    TBLIS_CONFIG_GEMM_KR(_,_,_,_)
    TBLIS_CONFIG_GEMM_MC(_,_,_,_)
    TBLIS_CONFIG_GEMM_NC(_,_,_,_)
    TBLIS_CONFIG_GEMM_KC(_,_,_,_)
    TBLIS_CONFIG_GEMM_UKR(_,_,_,_)
    TBLIS_CONFIG_GEMM_ROW_MAJOR(_,_,_,_)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_,_,_,_)

    TBLIS_CONFIG_PACK_NN_MR_UKR(_,_,_,_)
    TBLIS_CONFIG_PACK_NN_NR_UKR(_,_,_,_)
    TBLIS_CONFIG_PACK_SS_MR_UKR(_,_,_,_)
    TBLIS_CONFIG_PACK_SS_NR_UKR(_,_,_,_)
    //template <typename T> struct pack_nb_mr_ukr : static_microkernel<T, pack_nb_ukr_t, \
    //(pack_nb_ukr_def<this_config,   float,matrix_constants::MAT_A>), \
    //(pack_nb_ukr_def<this_config,  double,matrix_constants::MAT_A>), \
    //(pack_nb_ukr_def<this_config,scomplex,matrix_constants::MAT_A>), \
    //(pack_nb_ukr_def<this_config,dcomplex,matrix_constants::MAT_A>)> {};
    TBLIS_CONFIG_PACK_NB_MR_UKR(_,_,_,_)
    TBLIS_CONFIG_PACK_NB_NR_UKR(_,_,_,_)
    TBLIS_CONFIG_PACK_SS_SCAL_MR_UKR(_,_,_,_)
    TBLIS_CONFIG_PACK_SS_SCAL_NR_UKR(_,_,_,_)

    TBLIS_CONFIG_UPDATE_NN_UKR(_,_,_,_)
    TBLIS_CONFIG_UPDATE_SS_UKR(_,_,_,_)

    TBLIS_CONFIG_M_THREAD_RATIO(_,_,_,_)
    TBLIS_CONFIG_N_THREAD_RATIO(_,_,_,_)
    TBLIS_CONFIG_MR_MAX_THREAD(_,_,_,_)
    TBLIS_CONFIG_NR_MAX_THREAD(_,_,_,_)
};

}

#endif //TBLIS_INTERNAL_CONFIGS_HPP
