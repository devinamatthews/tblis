#ifndef _TCI_YIELD_H_
#define _TCI_YIELD_H_

#include "tci_config.hpp"

#if TCI_ARCH_MIC
#include <immintrin.h>
#elif TCI_ARCH_INTEL
#include <xmmintrin.h>
#endif

#if TCI_ARCH_MIC

static inline void tci_yield()
{
    _mm_delay(32);
}

#elif TCI_ARCH_INTEL

static inline void tci_yield()
{
    //_mm_pause();
    __asm__ __volatile__ ("pause");
}

#else

static inline void tci_yield() {}

#endif

#endif
