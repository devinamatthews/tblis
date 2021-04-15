#ifndef _TBLIS_CPUID_HPP_
#define _TBLIS_CPUID_HPP_

#include <cstring>
#include <cstdio>

#include "basic_types.h"

inline bool check_features(unsigned long have, unsigned long want)
{
    return (have&want) == want;
}

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

#include <cpuid.h>

namespace tblis
{

enum {VENDOR_INTEL, VENDOR_AMD, VENDOR_UNKNOWN};
enum : unsigned long {FEATURE_SSE3     = 0x0001u,
                      FEATURE_SSSE3    = 0x0002u,
                      FEATURE_SSE41    = 0x0004u,
                      FEATURE_SSE42    = 0x0008u,
                      FEATURE_AVX      = 0x0010u,
                      FEATURE_AVX2     = 0x0020u,
                      FEATURE_FMA3     = 0x0040u,
                      FEATURE_FMA4     = 0x0080u,
                      FEATURE_AVX512F  = 0x0100u,
                      FEATURE_AVX512PF = 0x0200u,
                      FEATURE_AVX512DQ = 0x0400u,
                      FEATURE_AVX512BW = 0x0800u,
                      FEATURE_AVX512VL = 0x1000u};

int get_cpu_type(int& family, int& model, int& features);

}

#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM)

namespace tblis
{

enum {VENDOR_ARM      = 0x41,
      VENDOR_BROADCOM = 0x42,
      VENDOR_CAVIUM   = 0x43,
      VENDOR_DEC      = 0x44,
      VENDOR_FUJITSU  = 0x45,
      VENDOR_NVIDIA   = 0x4e,
      VENDOR_APM      = 0x50,
      VENDOR_QUALCOMM = 0x51,
      VENDOR_SAMSUNG  = 0x53,
      VENDOR_TEXAS    = 0x54,
      VENDOR_MARVELL  = 0x56,
      VENDOR_UNKNOWN  = 0x00,
      VENDOR_APPLE    = 0xff}; // Apple does not have /proc/cpuinfo.
enum {MODEL_ARMV7, MODEL_ARMV8, MODEL_UNKNOWN};
enum {FEATURE_NEON, FEATURE_SVE};

int get_cpu_type(int& model, int& part, int& features);

}

#endif

#endif
