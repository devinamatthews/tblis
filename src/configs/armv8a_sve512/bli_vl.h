#ifndef _TBLIS_CONFIGS_ARMV8A_SVE512_BLI_VL_H_
#define _TBLIS_CONFIGS_ARMV8A_SVE512_BLI_VL_H_
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
uint64_t bli_vl()
{
    uint64_t vl_bytes = 0;
    __asm__ volatile(
            " mov %[vl_bytes],#0\n\t"
            " incb %[vl_bytes]\n\t"
            : [vl_bytes] "=r" (vl_bytes)
            :
            :
            ); 
    return vl_bytes;
}

#ifdef __cplusplus
}
#endif
#endif
