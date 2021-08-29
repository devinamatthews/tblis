#include "bli_vl.h"

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

