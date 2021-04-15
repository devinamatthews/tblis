#ifndef _TCI_YIELD_H_
#define _TCI_YIELD_H_

#include "tci_global.h"

#ifdef __cplusplus
extern "C" {
#endif

#if TCI_ARCH_MIC

TCI_INLINE void tci_yield()
{
    _mm_delay(32);
}

#elif TCI_ARCH_X86 || TCI_ARCH_X64

TCI_INLINE void tci_yield()
{
    //_mm_pause();
    __asm__ __volatile__ ("pause");
}

#else

TCI_INLINE void tci_yield() {}

#endif

#ifdef __cplusplus
}
#endif

#endif
