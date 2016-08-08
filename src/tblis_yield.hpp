#ifndef _TBLIS_YIELD_HPP_
#define _TBLIS_YIELD_HPP_

#include "tblis_config.hpp"

#if TBLIS_ARCH_MIC
#include <immintrin.h>
#elif TBLIS_ARCH_INTEL
#include <xmmintrin.h>
#endif

namespace tblis
{

#if TBLIS_ARCH_MIC

inline void yield()
{
    _mm_delay(32);
}

#elif TBLIS_ARCH_INTEL

inline void yield()
{
    //_mm_pause();
    __asm__ __volatile__ ("pause");
}

#else

inline void yield() {}

#endif

}

#endif
