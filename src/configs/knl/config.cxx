#include "util/cpuid.hpp"
#include "config.hpp"

#include "blis.h"

template <typename T>
using bli_packm_t = void(*)(conj_t conja, len_type n, const T* kappa,
                            const T* a, stride_type rs_a, stride_type cs_a,
                                  T* p,                   stride_type cs_p);

template <typename T>
using bli_packm_func = typename std::remove_pointer<bli_packm_t<T>>::type;

extern "C" bli_packm_func<double> bli_dpackm_30xk_opt;
extern "C" bli_packm_func<double> bli_dpackm_24xk_opt;
extern "C" bli_packm_func<double> bli_dpackm_8xk_opt;
extern "C" bli_packm_func<float> bli_spackm_24xk_opt;
extern "C" bli_packm_func<float> bli_spackm_16xk_opt;

namespace tblis
{

void knl_dpackm_30xk(len_type m, len_type k,
                     const double* p_a, stride_type rs_a, stride_type cs_a,
                     double* p_ap)
{
    constexpr double one = 1.0;

    if (m == 30)
    {
        bli_dpackm_30xk_opt(BLIS_NO_CONJUGATE, k, &one, p_a, rs_a, cs_a, p_ap, 32);
    }
    else
    {
        pack_nn_ukr_def<knl_d30x8_knc_config, double, matrix_constants::MAT_A>
            (m, k, p_a, rs_a, cs_a, p_ap);
    }
}

void knl_dpackm_24xk(len_type m, len_type k,
                     const double* p_a, stride_type rs_a, stride_type cs_a,
                     double* p_ap)
{
    constexpr double one = 1.0;

    if (m == 24)
    {
        bli_dpackm_24xk_opt(BLIS_NO_CONJUGATE, k, &one, p_a, rs_a, cs_a, p_ap, 24);
    }
    else
    {
        pack_nn_ukr_def<knl_d24x8_config, double, matrix_constants::MAT_A>
            (m, k, p_a, rs_a, cs_a, p_ap);
    }
}

void knl_dpackm_8xk(len_type m, len_type k,
                    const double* p_a, stride_type rs_a, stride_type cs_a,
                    double* p_ap)
{
    constexpr double one = 1.0;

    if (m == 8)
    {
        bli_dpackm_8xk_opt(BLIS_NO_CONJUGATE, k, &one, p_a, rs_a, cs_a, p_ap, 8);
    }
    else
    {
        pack_nn_ukr_def<knl_d24x8_config, double, matrix_constants::MAT_B>
            (m, k, p_a, rs_a, cs_a, p_ap);
    }
}

void knl_spackm_24xk(len_type m, len_type k,
                     const float* p_a, stride_type rs_a, stride_type cs_a,
                     float* p_ap)
{
    constexpr float one = 1.0;

    if (m == 24)
    {
        bli_spackm_24xk_opt(BLIS_NO_CONJUGATE, k, &one, p_a, rs_a, cs_a, p_ap, 24);
    }
    else
    {
        pack_nn_ukr_def<knl_d24x8_config, float, matrix_constants::MAT_A>
            (m, k, p_a, rs_a, cs_a, p_ap);
    }
}

void knl_spackm_16xk(len_type m, len_type k,
                     const float* p_a, stride_type rs_a, stride_type cs_a,
                     float* p_ap)
{
    constexpr float one = 1.0;

    if (m == 16)
    {
        bli_spackm_16xk_opt(BLIS_NO_CONJUGATE, k, &one, p_a, rs_a, cs_a, p_ap, 16);
    }
    else
    {
        pack_nn_ukr_def<knl_d24x8_config, float, matrix_constants::MAT_B>
            (m, k, p_a, rs_a, cs_a, p_ap);
    }
}

int knl_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_INTEL)
    {   
        if (get_verbose() >= 1) printf("tblis: knl: Wrong vendor.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX))
    {   
        if (get_verbose() >= 1) printf("tblis: knl: Doesn't support AVX.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_FMA3))
    {   
        if (get_verbose() >= 1) printf("tblis: knl: Doesn't support FMA3.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX2))
    {
        if (get_verbose() >= 1) printf("tblis: knl: Doesn't support AVX2.\n");
        return -1;
    }
    
    if (!check_features(features, FEATURE_AVX512F))
    {
        if (get_verbose() >= 1) printf("tblis: knl: Doesn't support AVX512F.\n");
        return -1;
    }
    
    if (!check_features(features, FEATURE_AVX512PF))
    {
        if (get_verbose() >= 1) printf("tblis: knl: Doesn't support AVX512PF.\n");
        return -1;
    }
    
    return 4;
}

//TBLIS_CONFIG_INSTANTIATE(knl_d30x8_knc);
//TBLIS_CONFIG_INSTANTIATE(knl_d30x8);
TBLIS_CONFIG_INSTANTIATE(knl_d24x8);
//TBLIS_CONFIG_INSTANTIATE(knl_d8x24);

}
