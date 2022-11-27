#include <tblis/internal/cpuid.hpp>
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

/*
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
*/

void knl_dpackm_24xk(len_type m, len_type k, const void* alpha, bool conj,
                     const void* p_a, stride_type rs_a, stride_type cs_a,
                     const void* p_d, stride_type inc_d,
                     const void* p_e, stride_type inc_e,
                     void* p_ap)
{
    if (m == 24 && !p_d && !p_e)
    {
        bli_dpackm_24xk_opt(conj ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,
            k, reinterpret_cast<const double*>(alpha),
            reinterpret_cast<const double*>(p_a), rs_a, cs_a,
            reinterpret_cast<double*>(p_ap), 24);
    }
    else
    {
        pack_nn_ukr_def<knl_config, double, matrix_constants::MAT_A>
            (m, k, alpha, conj, p_a, rs_a, cs_a, p_d, inc_d, p_e, inc_e, p_ap);
    }
}

void knl_dpackm_8xk(len_type m, len_type k, const void* alpha, bool conj,
                    const void* p_a, stride_type rs_a, stride_type cs_a,
                    const void* p_d, stride_type inc_d,
                    const void* p_e, stride_type inc_e,
                    void* p_ap)
{
    if (m == 8)
    {
        bli_dpackm_8xk_opt(conj ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,
            k, reinterpret_cast<const double*>(alpha),
            reinterpret_cast<const double*>(p_a), rs_a, cs_a,
            reinterpret_cast<double*>(p_ap), 8);
    }
    else
    {
        pack_nn_ukr_def<knl_config, double, matrix_constants::MAT_B>
            (m, k, alpha, conj, p_a, rs_a, cs_a, p_d, inc_d, p_e, inc_e, p_ap);
    }
}

void knl_spackm_24xk(len_type m, len_type k, const void* alpha, bool conj,
                     const void* p_a, stride_type rs_a, stride_type cs_a,
                     const void* p_d, stride_type inc_d,
                     const void* p_e, stride_type inc_e,
                     void* p_ap)
{
    if (m == 24)
    {
        bli_spackm_24xk_opt(conj ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,
            k, reinterpret_cast<const float*>(alpha),
            reinterpret_cast<const float*>(p_a), rs_a, cs_a,
            reinterpret_cast<float*>(p_ap), 24);
    }
    else
    {
        pack_nn_ukr_def<knl_config, float, matrix_constants::MAT_A>
            (m, k, alpha, conj, p_a, rs_a, cs_a, p_d, inc_d, p_e, inc_e, p_ap);
    }
}

void knl_spackm_16xk(len_type m, len_type k, const void* alpha, bool conj,
                     const void* p_a, stride_type rs_a, stride_type cs_a,
                     const void* p_d, stride_type inc_d,
                     const void* p_e, stride_type inc_e,
                     void* p_ap)
{
    if (m == 16)
    {
        bli_spackm_16xk_opt(conj ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,
            k, reinterpret_cast<const float*>(alpha),
            reinterpret_cast<const float*>(p_a), rs_a, cs_a,
            reinterpret_cast<float*>(p_ap), 16);
    }
    else
    {
        pack_nn_ukr_def<knl_config, float, matrix_constants::MAT_B>
            (m, k, alpha, conj, p_a, rs_a, cs_a, p_d, inc_d, p_e, inc_e, p_ap);
    }
}

TBLIS_CONFIG_INSTANTIATE(knl);

}
