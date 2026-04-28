#ifndef _TCI_YIELD_H_
#define _TCI_YIELD_H_

#include "tci/tci_config.h"

#if defined(__MIC__)
#define TCI_ARCH_MIC 1
#elif defined(__ia64) || defined(__itanium__) || defined(_M_IA64)
#define TCI_ARCH_IA64 1
#elif defined(__x86_64__) || defined(_M_X64)
#define TCI_ARCH_X64 1
#elif defined(__i386) || defined(_M_IX86)
#define TCI_ARCH_X86 1
#elif defined(__aarch64__)
#define TCI_ARCH_ARM64 1
#elif defined(__arm__) || defined(_M_ARM)
#define TCI_ARCH_ARM32 1
#elif defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
#define TCI_ARCH_PPC64 1
#elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__)
#define TCI_ARCH_PPC32 1
#elif defined(__bgq__)
#define TCI_ARCH_BGQ 1
#elif defined(__sparc)
#define TCI_ARCH_SPARC 1
#elif defined(__mips)
#define TCI_ARCH_MIPS 1
#else
#error "Unknown architecture"
#endif

#if TCI_ARCH_MIC
#include <immintrin.h>
#endif

#if TCI_ARCH_X86 || TCI_ARCH_X64
#include <xmmintrin.h>
#endif

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
