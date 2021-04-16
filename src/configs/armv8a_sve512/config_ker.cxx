#include "config.hpp"

#include "util/cpuid.hpp"

#define DPACKM_PARAMS \
     ( \
       conj_t        conja, \
       dim_t         n_,    \
       const double* kappa, \
       const double* a, inc_t inca_, inc_t lda_, \
       double*       p,              inc_t ldp_  \
     )
extern "C" void bli_dpackm_armsve512_asm_16xk_simp DPACKM_PARAMS;
extern "C" void bli_dpackm_armsve512_asm_10xk_simp DPACKM_PARAMS;

namespace tblis
{

void sve512_dpackm_asm_16xk(len_type m, len_type k,
                            const void* alpha, bool conj,
                            const void* p_a, stride_type rs_a, stride_type cs_a,
                            const void* p_d, stride_type inc_d,
                            const void* p_e, stride_type inc_e,
                            void* p_ap)
{
    int gs    = rs_a != 1 && cs_a != 1;
    int unitk = *((double *)alpha) == double(1.0);
    if (m == 16 && !gs && unitk)
    {
        bli_dpackm_armsve512_asm_16xk_simp(conj ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE, k,
                                           reinterpret_cast<const double*>(alpha),
                                           reinterpret_cast<const double*>(p_a), rs_a, cs_a,
                                           reinterpret_cast<double*>(p_ap), 16);
    }
    else
    {
        pack_nn_ukr_def<armv8a_sve512_config, double, matrix_constants::MAT_A>
            (m, k, alpha, conj, p_a, rs_a, cs_a, p_d, inc_d, p_e, inc_e, p_ap);
    }
}

void sve512_dpackm_asm_10xk(len_type m, len_type k,
                            const void* alpha, bool conj,
                            const void* p_a, stride_type rs_a, stride_type cs_a,
                            const void* p_d, stride_type inc_d,
                            const void* p_e, stride_type inc_e,
                            void* p_ap)
{
    int gs    = rs_a != 1 && cs_a != 1;
    int unitk = *((double *)alpha) == double(1.0);
    if (m == 10 && !gs && unitk)
    {
        bli_dpackm_armsve512_asm_10xk_simp(conj ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE, k,
                                           reinterpret_cast<const double*>(alpha),
                                           reinterpret_cast<const double*>(p_a), rs_a, cs_a,
                                           reinterpret_cast<double*>(p_ap), 10);
    }
    else
    {
        pack_nn_ukr_def<armv8a_sve512_config, double, matrix_constants::MAT_B>
            (m, k, alpha, conj, p_a, rs_a, cs_a, p_d, inc_d, p_e, inc_e, p_ap);
    }
}

TBLIS_CONFIG_INSTANTIATE(armv8a_sve512);

}
