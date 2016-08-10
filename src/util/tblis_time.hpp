#ifndef _TBLIS_TIME_HPP_
#define _TBLIS_TIME_HPP_

#ifdef __MACH__
#include <mach/mach_time.h>
#else
#include <ctime>
#endif

namespace tblis
{

inline double tic()
{
    #ifdef __MACH__
    static double conv = -1.0;
    if (conv < 0)
    {
        mach_timebase_info_data_t timebase;
        mach_timebase_info(&timebase);
        conv = (double)timebase.numer / (double)timebase.denom;
    }
    uint64_t nsec = mach_absolute_time();
    return conv*(double)nsec/1e9;
    #else
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec+(double)ts.tv_nsec/1e9;
    #endif
}

}

#endif
