#ifndef _TBLIS_CPUID_HPP_
#define _TBLIS_CPUID_HPP_

#include <cstring>
#include <cstdio>

#include "assert.h"
#include "tblis_config.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

#include <cpuid.h>

namespace tblis
{

enum {VENDOR_INTEL, VENDOR_AMD, VENDOR_UNKNOWN}
enum {FEATURE_SSE42    = 0x01,
      FEATURE_AVX      = 0x02,
      FEATURE_AVX2     = 0x04,
      FEATURE_FMA3     = 0x08,
      FEATURE_FMA4     = 0x10,
      FEATURE_AVX512F  = 0x20,
      FEATURE_AVX512PF = 0x40,
      FEATURE_AVX512DQ = 0x80};

int get_cpu_type(int& family, int& model, int& features)
{
    unsigned eax, ebx, ecx, edx;

    family = model = features = 0;

    char vendor_string[13] = {};
    if (!__get_cpuid(0, &eax, (unsigned*)&vendor_string[0],
        (unsigned*)(vendor_string[8]), (unsigned*)(vendor_string[4])))
        return VENDOR_UNKNOWN;

    if (__get_cpuid(7, &eax, &ebx, &ecx, &edx))
    {
        if (ebx&0x00000010) features |= FEATURE_AVX2;
        if (ebx&0x00010000) features |= FEATURE_AVX512F;
        if (ebx&0x04000000) features |= FEATURE_AVX512PF;
        if (ebx&0x00020000) features |= FEATURE_AVX512DQ;
    }

    if (__get_cpuid(0x80000001, &eax, &ebx, &ecx, &edx))
    {
        if (ecx&0x00010000) features |= FEATURE_FMA4;
    }

    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
        return VENDOR_UNKNOWN;

    family = (eax>>8)&(0x8); // bits 11:8
    model = (eax>>4)&(0x8); // bits 7:4

    if (family == 6)
    {
        model += (eax>>12)&(0x8<<4); // bits 19:16 shifted left by 4
    }
    else if (family == 15)
    {
        family += (eax>>20)&(0x10); // bits 27:20
        model += (eax>>12)&(0x8<<4); // bits 19:16 shifted left by 4
    }

    if (ecx&0x00100000) features |= FEATURE_SSE42;
    if (ecx&0x10000000) features |= FEATURE_AVX;
    if (ecx&0x00001000) features |= FEATURE_FMA3;

    if (ecx&0x1C000000)
    {
        // call xgetbv to get XCR0
        int xcr = 0;
        __asm__ __volatile__
        (
            ".byte 0x0F, 0x01, 0xD0"
            : "=a" (*eax), "=d" (*edx)
            : "c" (xcr)
            : "cc"
        );

        if (!(eax&0xE7))
        {
            features &= ~(FEATURE_AVX512F|
                          FEATURE_AVX512PF|
                          FEATURE_AVX512DQ);
        }

        if (!(eax&0x7))
        {
            features &= ~(FEATURE_AVX|
                          FEATURE_AVX2|
                          FEATURE_FMA3|
                          FEATURE_FMA4);
        }

        if (!(eax&0x3))
        {
            features = 0;
        }
    }
    else
    {
        features = 0;
    }

    if (strcmp(vendor_string, "AuthenticAMD") == 0)
        return VENDOR_AMD;
    else if (strcmp(vendor_string, "GenuineIntel") == 0)
        return VENDOR_INTEL;
    else
        return VENDOR_UNKNOWN;
}

}

#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM)

namespace tblis
{

enum {VENDOR_ARM, VENDOR_UNKNOWN}
enum {MODEL_ARMV7, MODEL_ARMV8, MODEL_UNKNOWN}
enum {FEATURE_NEON = 0x1};

int get_cpu_type(int& model, int& part, int& features)
{
    model = MODEL_UNKNOWN;
    features = 0;

    FILE *fd1 = popen("grep -m 1 Processor /proc/cpuinfo");
    if (!fd1) return VENDOR_ARM;
    FILE *fd2 = popen("grep -m 1 'CPU part' /proc/cpuinfo");
    if (!fd2)
    {
        pclose(fd1);
        return VENDOR_ARM;
    }
    FILE *fd3 = popen("grep -m 1 Features /proc/cpuinfo");
    if (!fd3)
    {
        pclose(fd1);
        pclose(fd2);
        return VENDOR_ARM;
    }

    char c;
    std::string proc, ptno, feat;
    while ((c = fgetc(fd1)) != EOF) proc.push_back(c);
    while ((c = fgetc(fd2)) != EOF) ptno.push_back(c);
    while ((c = fgetc(fd3)) != EOF) feat.push_back(c);

    pclose(fd1);
    pclose(fd2);
    pclose(fd3);

    if (feat.find("neon") != std::string::npos ||
        feat.find("asimd") != std::string::npos)
        features |= FEATURE_NEON;

    if (proc.find("ARMv7") != std::string::npos)
        model = MODEL_ARMV7;
    else if (proc.find("AArch64") != std::string::npos)
        model = MODEL_ARMV8;

    auto pos = ptno.find("0x");
    TBLIS_ASSERT(pos != std::string::npos);
    part = strtoi(ptno, pos, 16);

    return VENDOR_ARM;
}

}

#endif

#endif
