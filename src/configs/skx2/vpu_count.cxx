/*
 * Copyright 2017 Intel Corp.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <cpuid.h>

#ifdef USE_STRNSTR
#define STRSTR(haystack,needle) strnstr((haystack),(needle),sizeof(haystack))
#else
#define STRSTR(haystack,needle) strstr((haystack),(needle))
#endif

static void get_cpu_name(char cpu_name[48])
{
    uint32_t eax, ebx, ecx, edx;

    __cpuid(0x80000002u, eax, ebx, ecx, edx);
    //printf("%x %x %x %x\n", eax, ebx, ecx, edx);

    *(uint32_t *)&cpu_name[0]  = eax;
    *(uint32_t *)&cpu_name[4]  = ebx;
    *(uint32_t *)&cpu_name[8]  = ecx;
    *(uint32_t *)&cpu_name[12] = edx;

    __cpuid(0x80000003u, eax, ebx, ecx, edx);
    //printf("%x %x %x %x\n", eax, ebx, ecx, edx);

    *(uint32_t *)&cpu_name[16+0]  = eax;
    *(uint32_t *)&cpu_name[16+4]  = ebx;
    *(uint32_t *)&cpu_name[16+8]  = ecx;
    *(uint32_t *)&cpu_name[16+12] = edx;

    __cpuid(0x80000004u, eax, ebx, ecx, edx);
    //printf("%x %x %x %x\n", eax, ebx, ecx, edx);

    *(uint32_t *)&cpu_name[32+0]  = eax;
    *(uint32_t *)&cpu_name[32+4]  = ebx;
    *(uint32_t *)&cpu_name[32+8]  = ecx;
    *(uint32_t *)&cpu_name[32+12] = edx;
}

int vpu_count(void)
{
    char cpu_name[48] = {0};

    get_cpu_name(cpu_name);
#ifdef DEBUG
    printf("cpu_name = %s\n", cpu_name);
#endif

    const char skx[17]     = "Intel(R) Xeon(R)";
    const char knl[29]     = "Intel(R) Xeon Phi(TM) CPU 72";
    const char platinum[9] = "Platinum";
    const char gold[5]     = "Gold";
    const char silver[7]   = "Silver";
    const char bronze[7]   = "Bronze";

    const char * loc;

    /* Xeon Phi 72xx aka KNL always has 2 VPU */
    loc = STRSTR(cpu_name, knl);
    if (loc != NULL) {
        return 2;
    }

    /* FIXME: Add Core i9 X-series... */

    /* If it is not Xeon, it doesn't have AVX-512 */
    loc = STRSTR(cpu_name, skx);
    if (loc == NULL) {
        return 0;
    }

    loc = STRSTR(cpu_name, platinum);
    if (loc != NULL) {
        char skustr[5] = {0};
        memcpy(&skustr,loc+sizeof(platinum),4);
        const int skunum = atoi(skustr);
        if (8199 >= skunum && skunum >= 8100) {
            return 2;
        } else {
            printf("%s %d does not exist.\n", platinum, skunum);
        }
    }

    loc = STRSTR(cpu_name, gold);
    if (loc != NULL) {
        char skustr[5] = {0};
        memcpy(&skustr,loc+sizeof(gold),4);
        const int skunum = atoi(skustr);
        if (skunum == 5122) {
            /* https://ark.intel.com/products/120475/Intel-Xeon-Gold-5122-Processor-16_5M-Cache-3_60-GHz */
            return 2;
        } else if (6199 >= skunum && skunum >= 6100) {
            return 2;
        } else if (5199 >= skunum && skunum >= 5100) {
            return 1;
        } else {
            printf("%s %d does not exist.\n", gold, skunum);
        }
    }

    loc = STRSTR(cpu_name, silver);
    if (loc != NULL) {
        char skustr[5] = {0};
        memcpy(&skustr,loc+sizeof(silver),4);
        const int skunum = atoi(skustr);
        if (4199 >= skunum && skunum >= 4100) {
            return 1;
        } else {
            printf("%s %d does not exist.\n", silver, skunum);
        }
    }

    loc = STRSTR(cpu_name, bronze);
    if (loc != NULL) {
        char skustr[5] = {0};
        memcpy(&skustr,loc+sizeof(bronze),4);
        const int skunum = atoi(skustr);
        if (3199 >= skunum && skunum >= 3100) {
            return 1;
        } else {
            printf("%s %d does not exist.\n", bronze, skunum);
        }
    }

    return 0;
}
