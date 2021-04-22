#ifndef _TCI_GLOBAL_H_
#define _TCI_GLOBAL_H_

#include "tci_config.h"

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

#define TCI_MIN(x,y) ((y)<(x)?(y):(x))
#define TCI_MAX(x,y) ((x)<(y)?(y):(x))

#include <stdint.h>
#include <errno.h>

#ifdef __cplusplus
#define TCI_INLINE inline
#else
#define TCI_INLINE static inline
#include <stdbool.h>
#endif

#if TCI_ARCH_MIC
#include <immintrin.h>
#endif

#if TCI_ARCH_X86 || TCI_ARCH_X64
#include <xmmintrin.h>
#endif

#if TCI_USE_OS_UNFAIR_LOCK
#include <os/lock.h>
#endif

#if TCI_USE_OSX_SPINLOCK
#include <libkern/OSAtomic.h>
#endif

#if TCI_USE_PTHREAD_SPINLOCK || \
    TCI_USE_PTHREAD_MUTEX || \
    TCI_USE_PTHREAD_BARRIER || \
    TCI_USE_PTHREADS_THREADS
#include <pthread.h>
#endif

#if TCI_USE_OMP_LOCK || TCI_USE_OPENMP_THREADS
#include <omp.h>
#endif

#if TCI_USE_TBB_THREADS && defined(__cplusplus)
#include <tbb/tbb.h>
#endif

#if TCI_USE_DISPATCH_THREADS
#include <dispatch/dispatch.h>
#endif

#if TCI_USE_PPL_THREADS && defined(__cplusplus)
#include <ppl.h>
#endif

#if TCI_USE_WINDOWS_THREADS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#endif
