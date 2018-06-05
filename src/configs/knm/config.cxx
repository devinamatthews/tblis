#include "util/cpuid.hpp"
#include "config.hpp"

#include "blis.h"

namespace tblis
{

void knm_spackm_24xk(len_type m, len_type k,
                     const float* p_a, stride_type rs_a, stride_type cs_a,
                     float* p_ap)
{
    constexpr len_type MR = 24;
    constexpr len_type KR = 4;

    if (m == MR && rs_a == 1)
    {
        len_type p = 0;
        for (;p <= k-KR;p += KR)
        {
            for (len_type mr = 0;mr < MR;mr++)
            {
                for (len_type kr = 0;kr < KR;kr++)
                {
                    p_ap[mr*KR + kr] = p_a[mr + cs_a*kr];
                }
            }

            p_a += cs_a*KR;
            p_ap += MR*KR;
        }

        for (len_type kr = 0;kr < k-p;kr++)
        {
            for (len_type mr = 0;mr < MR;mr++)
            {
                p_ap[mr + MR*kr] = p_a[mr + cs_a*kr];
            }
        }
    }
    else if (m == MR && cs_a == 1)
    {
        len_type p = 0;
        for (;p <= k-KR;p += KR)
        {
            for (len_type mr = 0;mr < MR;mr++)
            {
                for (len_type kr = 0;kr < KR;kr++)
                {
                    p_ap[mr*KR + kr] = p_a[rs_a*mr + kr];
                }
            }

            p_a += KR;
            p_ap += MR*KR;
        }

        for (len_type kr = 0;kr < k-p;kr++)
        {
            for (len_type mr = 0;mr < MR;mr++)
            {
                p_ap[mr + MR*kr] = p_a[rs_a*mr + kr];
            }
        }
    }
    else
    {
        len_type p = 0;
        for (;p <= k-KR;p += KR)
        {
            for (len_type mr = 0;mr < m;mr++)
            {
                for (len_type kr = 0;kr < KR;kr++)
                {
                    p_ap[mr*KR + kr] = p_a[rs_a*mr + cs_a*kr];
                }
            }

            for (len_type mr = m;mr < MR;mr++)
            {
                for (len_type kr = 0;kr < KR;kr++)
                {
                    p_ap[mr*KR + kr] = 0;
                }
            }

            p_a += cs_a*KR;
            p_ap += MR*KR;
        }

        for (len_type kr = 0;kr < k-p;kr++)
        {
            for (len_type mr = 0;mr < m;mr++)
            {
                p_ap[mr + MR*kr] = p_a[rs_a*mr + cs_a*kr];
            }

            for (len_type mr = m;mr < MR;mr++)
            {
                p_ap[mr + MR*kr] = 0;
            }
        }
    }
}

int knm_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_INTEL)
    {   
        if (get_verbose() >= 1) printf("tblis: knm: Wrong vendor.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX))
    {   
        if (get_verbose() >= 1) printf("tblis: knm: Doesn't support AVX.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_FMA3))
    {   
        if (get_verbose() >= 1) printf("tblis: knm: Doesn't support FMA3.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX2))
    {
        if (get_verbose() >= 1) printf("tblis: knm: Doesn't support AVX2.\n");
        return -1;
    }
    
    if (!check_features(features, FEATURE_AVX512F))
    {
        if (get_verbose() >= 1) printf("tblis: knm: Doesn't support AVX512F.\n");
        return -1;
    }
    
    if (!check_features(features, FEATURE_AVX512PF))
    {
        if (get_verbose() >= 1) printf("tblis: knm: Doesn't support AVX512PF.\n");
        return -1;
    }
    
    if (!check_features(features, FEATURE_AVX5124FMAPS))
    {
        if (get_verbose() >= 1) printf("tblis: knm: Doesn't support AVX5124FMAPS.\n");
        return -1;
    }

    return 5;
}

TBLIS_CONFIG_INSTANTIATE(knm_s24x16);
TBLIS_CONFIG_INSTANTIATE(knm_s16x24);

}
