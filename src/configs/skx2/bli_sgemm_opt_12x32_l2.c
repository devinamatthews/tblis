/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
   OF TEXAS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
   OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#include <assert.h>

#include "../../util/asm_x86.h"
#include "common.h"

#define CACHELINE_SIZE 64 //size of cache line in bytes

/* During each subiteration, prefetching 2 cache lines of B
 * UNROLL factor ahead. 2cache lines = 32 floats (NR).
 * */
#define PREFETCH_B_L1(n, k) \
    PREFETCH(0, MEM(RBX, B_L1_PREFETCH_DIST*32*4 + (2*n+k)  * CACHELINE_SIZE))

#define LOOP_ALIGN ALIGN16

#define UPDATE_C(R1,R2,R3,R4) \
\
    VMULPS(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPS(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPS(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPS(ZMM(R4), ZMM(R4), ZMM(0)) \
    VFMADD231PS(ZMM(R1), ZMM(1), MEM(RCX,0*64)) \
    VFMADD231PS(ZMM(R2), ZMM(1), MEM(RCX,1*64)) \
    VFMADD231PS(ZMM(R3), ZMM(1), MEM(RCX,RAX,1,0*64)) \
    VFMADD231PS(ZMM(R4), ZMM(1), MEM(RCX,RAX,1,1*64)) \
    VMOVUPS(MEM(RCX,0*64), ZMM(R1)) \
    VMOVUPS(MEM(RCX,1*64), ZMM(R2)) \
    VMOVUPS(MEM(RCX,RAX,1,0*64), ZMM(R3)) \
    VMOVUPS(MEM(RCX,RAX,1,1*64), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,2))

#define UPDATE_C_BZ(R1,R2,R3,R4) \
\
    VMULPS(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPS(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPS(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPS(ZMM(R4), ZMM(R4), ZMM(0)) \
    VMOVUPS(MEM(RCX,0*64), ZMM(R1)) \
    VMOVUPS(MEM(RCX,1*64), ZMM(R2)) \
    VMOVUPS(MEM(RCX,RAX,1,0*64), ZMM(R3)) \
    VMOVUPS(MEM(RCX,RAX,1,1*64), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,2))

#define UPDATE_C_BZ_RS_ONE(R,C) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPS(ZMM(R), ZMM(R), ZMM(0)) \
    VMOVAPS(ZMM(4), ZMM(R)) \
    VEXTRACTF64X4(YMM(5), ZMM(R), IMM(1)) \
    VSCATTERQPS(MEM(C,ZMM(2),1) MASK_K(1), YMM(4)) \
    VSCATTERQPS(MEM(C,ZMM(3),1) MASK_K(2), YMM(5))

#define UPDATE_C_RS_ONE(R,C) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    KXNORW(K(3), K(0), K(0)) \
    KXNORW(K(4), K(0), K(0)) \
    VMULPS(ZMM(R), ZMM(R), ZMM(0)) \
    VMOVAPS(ZMM(4), ZMM(R)) \
    VEXTRACTF64X4(YMM(5), ZMM(R), IMM(1)) \
    VGATHERQPS(YMM(6) MASK_K(1), MEM(C,ZMM(2),1)) \
    VGATHERQPS(YMM(7) MASK_K(2), MEM(C,ZMM(3),1)) \
    VFMADD231PS(YMM(4), YMM(6), YMM(1)) \
    VFMADD231PS(YMM(5), YMM(7), YMM(1)) \
    VSCATTERQPS(MEM(C,ZMM(2),1) MASK_K(3), YMM(4)) \
    VSCATTERQPS(MEM(C,ZMM(3),1) MASK_K(4), YMM(5))

#define UPDATE_C_ROW_SCATTERED(R1,R2,R3,R4) \
\
    UPDATE_C_RS_ONE(R1,RCX) \
    UPDATE_C_RS_ONE(R2,RDX) \
\
    LEA(RCX, MEM(RCX,RAX,1)) \
    LEA(RDX, MEM(RDX,RAX,1)) \
\
    UPDATE_C_RS_ONE(R3,RCX) \
    UPDATE_C_RS_ONE(R4,RDX) \
\
    LEA(RCX, MEM(RCX,RAX,1)) \
    LEA(RDX, MEM(RDX,RAX,1))

#define UPDATE_C_BZ_ROW_SCATTERED(R1,R2,R3,R4) \
\
    UPDATE_C_BZ_RS_ONE(R1,RCX) \
    UPDATE_C_BZ_RS_ONE(R2,RDX) \
\
    LEA(RCX, MEM(RCX,RAX,1)) \
    LEA(RDX, MEM(RDX,RAX,1)) \
\
    UPDATE_C_BZ_RS_ONE(R3,RCX) \
    UPDATE_C_BZ_RS_ONE(R4,RDX) \
\
    LEA(RCX, MEM(RCX,RAX,1)) \
    LEA(RDX, MEM(RDX,RAX,1))

#define UPDATE_C_TRANS16X12(Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9,Z10,Z11,Z12) \
\
    VMULPD(ZMM( Z1), ZMM( Z1), ZMM(0))                          \
    VMULPD(ZMM( Z2), ZMM( Z2), ZMM(0))                          \
    VMULPD(ZMM( Z3), ZMM( Z3), ZMM(0))                          \
    VMULPD(ZMM( Z4), ZMM( Z4), ZMM(0))                          \
    VMULPD(ZMM( Z5), ZMM( Z5), ZMM(0))                          \
    VMULPD(ZMM( Z6), ZMM( Z6), ZMM(0))                          \
    VMULPD(ZMM( Z7), ZMM( Z7), ZMM(0))                          \
    VMULPD(ZMM( Z8), ZMM( Z8), ZMM(0))                          \
    VMULPD(ZMM( Z9), ZMM( Z9), ZMM(0))                          \
    VMULPD(ZMM(Z10), ZMM(Z10), ZMM(0))                          \
    VMULPD(ZMM(Z11), ZMM(Z11), ZMM(0))                          \
    VMULPD(ZMM(Z12), ZMM(Z12), ZMM(0))                          \
    /* 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F -  Z1 */ \
    /* 10 11 12 13 14 15 16 17 18 19 1A 1B 1C 1D 1E 1F -  Z2 */ \
    /* 20 21 22 23 24 25 26 27 28 29 2A 2B 2C 2D 2E 2F -  Z3 */ \
    /* 30 31 32 33 34 35 36 37 38 39 3A 3B 3C 3D 3E 3F -  Z4 */ \
    /* 40 41 42 43 44 45 46 47 48 49 4A 4B 4C 4D 4E 4F -  Z5 */ \
    /* 50 51 52 53 54 55 56 57 58 59 5A 5B 5C 5D 5E 5F -  Z6 */ \
    /* 60 61 62 63 64 65 66 67 68 69 6A 6B 6C 6D 6E 6F -  Z7 */ \
    /* 70 71 72 73 74 75 76 77 78 79 7A 7B 7C 7D 7E 7F -  Z8 */ \
    /* 80 81 82 83 84 85 86 87 88 89 8A 8B 8C 8D 8E 8F -  Z9 */ \
    /* 90 91 92 93 94 95 96 97 98 99 9A 9B 9C 9D 9E 9F - Z10 */ \
    /* A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 AA AB AC AD AE AF - Z11 */ \
    /* B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 BA BB BC BD BE BF - Z12 */ \
    VUNPCKLPS(ZMM(  2), ZMM( Z1), ZMM( Z2))                     \
    VUNPCKHPS(ZMM( Z2), ZMM( Z1), ZMM( Z2))                     \
    VUNPCKLPS(ZMM(  3), ZMM( Z3), ZMM( Z4))                     \
    VUNPCKHPS(ZMM( Z4), ZMM( Z3), ZMM( Z4))                     \
    VUNPCKLPS(ZMM(  4), ZMM( Z5), ZMM( Z6))                     \
    VUNPCKHPS(ZMM( Z6), ZMM( Z5), ZMM( Z6))                     \
    VUNPCKLPS(ZMM(  5), ZMM( Z7), ZMM( Z8))                     \
    VUNPCKHPS(ZMM( Z8), ZMM( Z7), ZMM( Z8))                     \
    VUNPCKLPS(ZMM(  6), ZMM( Z9), ZMM(Z10))                     \
    VUNPCKHPS(ZMM(Z10), ZMM( Z9), ZMM(Z10))                     \
    VUNPCKHPS(ZMM(  7), ZMM(Z11), ZMM(Z12))                     \
    VUNPCKLPS(ZMM(Z11), ZMM(Z11), ZMM(Z12))                     \
    /* 00 10 02 12 04 14 06 16 08 18 0A 1A 0C 1C 0E 1E -   2 */ \
    /* 01 11 03 13 05 15 07 17 09 19 0B 1B 0D 1D 0F 1F -  Z2 */ \
    /* 20 30 22 32 24 34 26 36 28 38 2A 3A 2C 3C 2E 3E -   3 */ \
    /* 21 31 23 33 25 35 27 37 29 39 2B 3B 2D 3D 2F 3F -  Z4 */ \
    /* 40 50 42 52 44 54 46 56 48 58 4A 5A 4C 5C 4E 5E -   4 */ \
    /* 41 51 43 53 45 55 47 57 49 59 4B 5B 4D 5D 4F 5F -  Z6 */ \
    /* 60 70 62 72 64 74 66 76 68 78 6A 7A 6C 7C 6E 7E -   5 */ \
    /* 61 71 63 73 65 75 67 77 69 79 6B 7B 6D 7D 6F 7F -  Z8 */ \
    /* 80 90 82 92 84 94 86 96 88 98 8A 8A 8C 9C 8E 9E -   6 */ \
    /* 81 91 83 93 85 95 87 97 89 99 8B 9B 8D 9D 8F 9F - Z10 */ \
    /* A0 B0 A2 B2 A4 B4 A6 B6 A8 B8 AA BA AC BC AE BE - Z11 */ \
    /* A1 B1 A3 B3 A5 B5 A7 B7 A9 B9 AB BB AD BD AF BF -   7 */ \
    VSHUFPS(ZMM( Z3), ZMM(  2), ZMM(  3), IMM(0xee))            \
    VSHUFPS(ZMM(  2), ZMM(  2), ZMM(  3), IMM(0x44))            \
    VSHUFPS(ZMM(  3), ZMM( Z2), ZMM( Z4), IMM(0xee))            \
    VSHUFPS(ZMM( Z2), ZMM( Z2), ZMM( Z4), IMM(0x44))            \
    VSHUFPS(ZMM( Z5), ZMM(  4), ZMM(  5), IMM(0x44))            \
    VSHUFPS(ZMM(  5), ZMM(  4), ZMM(  5), IMM(0xee))            \
    VSHUFPS(ZMM(  4), ZMM( Z6), ZMM( Z8), IMM(0x44))            \
    VSHUFPS(ZMM( Z8), ZMM( Z6), ZMM( Z8), IMM(0xee))            \
    VSHUFPS(ZMM( Z9), ZMM(  6), ZMM(Z11), IMM(0x44))            \
    VSHUFPS(ZMM(Z11), ZMM(  6), ZMM(Z11), IMM(0xee))            \
    VSHUFPS(ZMM(Z12), ZMM(Z10), ZMM(  7), IMM(0xee))            \
    VSHUFPS(ZMM(Z10), ZMM(Z10), ZMM(  7), IMM(0x44))            \
    /* 00 10 20 30 04 14 24 34 08 18 28 38 0C 1C 2C 3C -   2 */ \
    /* 01 11 03 31 05 15 25 35 09 19 29 39 0D 1D 2D 3D -  Z2 */ \
    /* 02 12 22 32 06 16 26 36 0A 1A 2A 3A 0E 1E 2E 3E -  Z3 */ \
    /* 03 13 23 33 07 17 27 37 0B 1B 2B 3B 0F 1F 2F 3F -   3 */ \
    /* 40 50 60 70 44 54 64 74 48 58 68 78 4C 5C 6C 7C -  Z5 */ \
    /* 41 51 61 71 45 55 65 75 49 59 69 79 4D 5D 6D 7D -   4 */ \
    /* 42 52 62 72 46 56 66 76 4A 5A 6A 7A 4E 5E 6E 7E -   5 */ \
    /* 43 53 63 73 47 57 67 77 4B 5B 6B 7B 4F 5F 6F 7F -  Z8 */ \
    /* 80 90 A0 B0 84 94 A4 B4 88 98 A8 B8 8C 9C AC BC -  Z9 */ \
    /* 81 91 A1 B1 85 95 A5 B5 89 99 A9 B9 8D 9D AD BD - Z10 */ \
    /* 82 92 A2 B2 86 96 A6 B6 8A 9A AA BA 8E 9E AE BE - Z11 */ \
    /* 83 93 A3 B3 87 97 A7 B7 8B 9B AB BB 8F 9F AF BF - Z12 */ \
    VSHUFF32X4(ZMM( Z1), ZMM(  2), ZMM( Z5), IMM(0x44))         \
    VSHUFF32X4(ZMM( Z5), ZMM(  2), ZMM( Z5), IMM(0xee))         \
    VSHUFF32X4(ZMM( Z6), ZMM( Z2), ZMM(  4), IMM(0xee))         \
    VSHUFF32X4(ZMM( Z2), ZMM( Z2), ZMM(  4), IMM(0x44))         \
    VSHUFF32X4(ZMM( Z7), ZMM( Z3), ZMM(  5), IMM(0xee))         \
    VSHUFF32X4(ZMM( Z3), ZMM( Z3), ZMM(  5), IMM(0x44))         \
    VSHUFF32X4(ZMM( Z4), ZMM(  3), ZMM( Z8), IMM(0x44))         \
    VSHUFF32X4(ZMM( Z8), ZMM(  3), ZMM( Z8), IMM(0xee))         \
    /* 00 10 20 30 04 14 24 34 40 50 60 70 44 54 64 74 -  Z1 */ \
    /* 01 11 03 31 05 15 25 35 41 51 61 71 45 55 65 75 -  Z2 */ \
    /* 02 12 22 32 06 16 26 36 42 52 62 72 46 56 66 76 -  Z3 */ \
    /* 03 13 23 33 07 17 27 37 43 53 63 73 47 57 67 77 -  Z4 */ \
    /* 08 18 28 38 0C 1C 2C 3C 48 58 68 78 4C 5C 6C 7C -  Z5 */ \
    /* 09 19 29 39 0D 1D 2D 3D 49 59 69 79 4D 5D 6D 7D -  Z6 */ \
    /* 0A 1A 2A 3A 0E 1E 2E 3E 4A 5A 6A 7A 4E 5E 6E 7E -  Z7 */ \
    /* 0B 1B 2B 3B 0F 1F 2F 3F 4B 5B 6B 7B 4F 5F 6F 7F -  Z8 */ \
    /* 80 90 A0 B0 84 94 A4 B4 88 98 A8 B8 8C 9C AC BC -  Z9 */ \
    /* 81 91 A1 B1 85 95 A5 B5 89 99 A9 B9 8D 9D AD BD - Z10 */ \
    /* 82 92 A2 B2 86 96 A6 B6 8A 9A AA BA 8E 9E AE BE - Z11 */ \
    /* 83 93 A3 B3 87 97 A7 B7 8B 9B AB BB 8F 9F AF BF - Z12 */ \
    VSHUFF32X4(ZMM(  2), ZMM(Z1), ZMM( Z9), IMM(0x08))          \
    VSHUFF32X4(ZMM(  3), ZMM(Z2), ZMM(Z10), IMM(0x08))          \
    VSHUFF32X4(ZMM(  4), ZMM(Z3), ZMM(Z11), IMM(0x08))          \
    VSHUFF32X4(ZMM(  5), ZMM(Z4), ZMM(Z12), IMM(0x5d))          \
    VSHUFF32X4(ZMM(  6), ZMM(Z1), ZMM( Z9), IMM(0x5d))          \
    VSHUFF32X4(ZMM(  7), ZMM(Z2), ZMM(Z10), IMM(0x5d))          \
    VSHUFF32X4(ZMM( Z1), ZMM(Z3), ZMM(Z11), IMM(0x5d))          \
    VSHUFF32X4(ZMM( Z2), ZMM(Z4), ZMM(Z12), IMM(0x5d))          \
    VSHUFF32X4(ZMM( Z3), ZMM(Z5), ZMM( Z9), IMM(0xa8))          \
    VSHUFF32X4(ZMM( Z4), ZMM(Z6), ZMM(Z10), IMM(0xa8))          \
    VSHUFF32X4(ZMM( Z7), ZMM(Z7), ZMM(Z11), IMM(0xa8))          \
    VSHUFF32X4(ZMM( Z8), ZMM(Z8), ZMM(Z12), IMM(0xa8))          \
    VSHUFF32X4(ZMM( Z9), ZMM(Z5), ZMM( Z9), IMM(0xfd))          \
    VSHUFF32X4(ZMM(Z10), ZMM(Z6), ZMM(Z10), IMM(0xfd))          \
    VSHUFF32X4(ZMM(Z11), ZMM(Z7), ZMM(Z11), IMM(0xfd))          \
    VSHUFF32X4(ZMM(Z12), ZMM(Z8), ZMM(Z12), IMM(0xfd))          \
    /* 00 10 20 30 40 50 60 70 80 90 A0 B0 80 90 A0 B0 -   2 */ \
    /* 01 11 03 31 41 51 61 71 81 91 A1 B1 81 91 A1 B1 -   3 */ \
    /* 02 12 22 32 42 52 62 72 82 92 A2 B2 82 92 A2 B2 -   4 */ \
    /* 03 13 23 33 43 53 63 73 83 93 A3 B3 83 93 A3 B3 -   5 */ \
    /* 04 14 24 34 44 54 64 74 84 94 A4 B4 84 94 A4 B4 -   6 */ \
    /* 05 15 25 35 45 55 65 75 85 95 A5 B5 85 95 A5 B5 -   7 */ \
    /* 06 16 26 36 46 56 66 76 86 96 A6 B6 86 96 A6 B6 -  Z1 */ \
    /* 07 17 27 37 47 57 67 77 87 97 A7 B7 87 97 A7 B7 -  Z2 */ \
    /* 08 18 28 38 48 58 68 78 88 98 A8 B8 88 98 A8 B8 -  Z3 */ \
    /* 09 19 09 39 49 59 69 79 89 99 A9 B9 89 99 A9 B9 -  Z4 */ \
    /* 0A 1A 2A 3A 4A 5A 6A 7A 8A 9A AA BA 8A 9A AA BA -  Z7 */ \
    /* 0B 1B 2B 3B 4B 5B 6B 7B 8B 9B AB BB 8B 9B AB BB -  Z8 */ \
    /* 0C 1C 2C 3C 4C 5C 6C 7C 8C 9C AC BC 8C 9C AC BC -  Z9 */ \
    /* 0D 1D 2D 3D 4D 5D 6D 7D 8D 9D AD BD 8D 9D AD BD - Z10 */ \
    /* 0E 1E 2E 3E 4E 5E 6E 7E 8E 9E AE BE 8E 9E AE BE - Z11 */ \
    /* 0F 1F 2F 3F 4F 5F 6F 7F 8F 9F AF BF 8F 9F AF BF - Z12 */ \
    VFMADD231PS(ZMM(  2) MASK_K(1), ZMM(1), MEM(RCX      ))     \
    VFMADD231PS(ZMM(  3) MASK_K(1), ZMM(1), MEM(RCX,RBX,1))     \
    VFMADD231PS(ZMM(  4) MASK_K(1), ZMM(1), MEM(RCX,RBX,2))     \
    VFMADD231PS(ZMM(  5) MASK_K(1), ZMM(1), MEM(RCX,R13,1))     \
    VFMADD231PS(ZMM(  6) MASK_K(1), ZMM(1), MEM(RCX,RBX,4))     \
    VFMADD231PS(ZMM(  7) MASK_K(1), ZMM(1), MEM(RCX,R15,1))     \
    VFMADD231PS(ZMM( Z1) MASK_K(1), ZMM(1), MEM(RCX,R13,2))     \
    VFMADD231PS(ZMM( Z2) MASK_K(1), ZMM(1), MEM(RCX,R10,1))     \
    VFMADD231PS(ZMM( Z3) MASK_K(1), ZMM(1), MEM(RDX      ))     \
    VFMADD231PS(ZMM( Z4) MASK_K(1), ZMM(1), MEM(RDX,RBX,1))     \
    VFMADD231PS(ZMM( Z7) MASK_K(1), ZMM(1), MEM(RDX,RBX,2))     \
    VFMADD231PS(ZMM( Z8) MASK_K(1), ZMM(1), MEM(RDX,R13,1))     \
    VFMADD231PS(ZMM( Z9) MASK_K(1), ZMM(1), MEM(RDX,RBX,4))     \
    VFMADD231PS(ZMM(Z10) MASK_K(1), ZMM(1), MEM(RDX,R15,1))     \
    VFMADD231PS(ZMM(Z11) MASK_K(1), ZMM(1), MEM(RDX,R13,2))     \
    VFMADD231PS(ZMM(Z12) MASK_K(1), ZMM(1), MEM(RDX,R10,1))     \
    VMOVUPS(MEM(RCX      ) MASK_K(1), ZMM(  2))                 \
    VMOVUPS(MEM(RCX,RBX,1) MASK_K(1), ZMM(  3))                 \
    VMOVUPS(MEM(RCX,RBX,2) MASK_K(1), ZMM(  4))                 \
    VMOVUPS(MEM(RCX,R13,1) MASK_K(1), ZMM(  5))                 \
    VMOVUPS(MEM(RCX,RBX,4) MASK_K(1), ZMM(  6))                 \
    VMOVUPS(MEM(RCX,R15,1) MASK_K(1), ZMM(  7))                 \
    VMOVUPS(MEM(RCX,R13,2) MASK_K(1), ZMM( Z1))                 \
    VMOVUPS(MEM(RCX,R10,1) MASK_K(1), ZMM( Z2))                 \
    VMOVUPS(MEM(RDX      ) MASK_K(1), ZMM( Z3))                 \
    VMOVUPS(MEM(RDX,RBX,1) MASK_K(1), ZMM( Z4))                 \
    VMOVUPS(MEM(RDX,RBX,2) MASK_K(1), ZMM( Z7))                 \
    VMOVUPS(MEM(RDX,R13,1) MASK_K(1), ZMM( Z8))                 \
    VMOVUPS(MEM(RDX,RBX,4) MASK_K(1), ZMM( Z9))                 \
    VMOVUPS(MEM(RDX,R15,1) MASK_K(1), ZMM(Z10))                 \
    VMOVUPS(MEM(RDX,R13,2) MASK_K(1), ZMM(Z11))                 \
    VMOVUPS(MEM(RDX,R10,1) MASK_K(1), ZMM(Z12))

#define UPDATE_C_TRANS16X12_BZ(Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9,Z10,Z11,Z12) \
\
    VMULPD(ZMM( Z1), ZMM( Z1), ZMM(0))                          \
    VMULPD(ZMM( Z2), ZMM( Z2), ZMM(0))                          \
    VMULPD(ZMM( Z3), ZMM( Z3), ZMM(0))                          \
    VMULPD(ZMM( Z4), ZMM( Z4), ZMM(0))                          \
    VMULPD(ZMM( Z5), ZMM( Z5), ZMM(0))                          \
    VMULPD(ZMM( Z6), ZMM( Z6), ZMM(0))                          \
    VMULPD(ZMM( Z7), ZMM( Z7), ZMM(0))                          \
    VMULPD(ZMM( Z8), ZMM( Z8), ZMM(0))                          \
    VMULPD(ZMM( Z9), ZMM( Z9), ZMM(0))                          \
    VMULPD(ZMM(Z10), ZMM(Z10), ZMM(0))                          \
    VMULPD(ZMM(Z11), ZMM(Z11), ZMM(0))                          \
    VMULPD(ZMM(Z12), ZMM(Z12), ZMM(0))                          \
    /* 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F -  Z1 */ \
    /* 10 11 12 13 14 15 16 17 18 19 1A 1B 1C 1D 1E 1F -  Z2 */ \
    /* 20 21 22 23 24 25 26 27 28 29 2A 2B 2C 2D 2E 2F -  Z3 */ \
    /* 30 31 32 33 34 35 36 37 38 39 3A 3B 3C 3D 3E 3F -  Z4 */ \
    /* 40 41 42 43 44 45 46 47 48 49 4A 4B 4C 4D 4E 4F -  Z5 */ \
    /* 50 51 52 53 54 55 56 57 58 59 5A 5B 5C 5D 5E 5F -  Z6 */ \
    /* 60 61 62 63 64 65 66 67 68 69 6A 6B 6C 6D 6E 6F -  Z7 */ \
    /* 70 71 72 73 74 75 76 77 78 79 7A 7B 7C 7D 7E 7F -  Z8 */ \
    /* 80 81 82 83 84 85 86 87 88 89 8A 8B 8C 8D 8E 8F -  Z9 */ \
    /* 90 91 92 93 94 95 96 97 98 99 9A 9B 9C 9D 9E 9F - Z10 */ \
    /* A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 AA AB AC AD AE AF - Z11 */ \
    /* B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 BA BB BC BD BE BF - Z12 */ \
    VUNPCKLPS(ZMM(  2), ZMM( Z1), ZMM( Z2))                     \
    VUNPCKHPS(ZMM( Z2), ZMM( Z1), ZMM( Z2))                     \
    VUNPCKLPS(ZMM(  3), ZMM( Z3), ZMM( Z4))                     \
    VUNPCKHPS(ZMM( Z4), ZMM( Z3), ZMM( Z4))                     \
    VUNPCKLPS(ZMM(  4), ZMM( Z5), ZMM( Z6))                     \
    VUNPCKHPS(ZMM( Z6), ZMM( Z5), ZMM( Z6))                     \
    VUNPCKLPS(ZMM(  5), ZMM( Z7), ZMM( Z8))                     \
    VUNPCKHPS(ZMM( Z8), ZMM( Z7), ZMM( Z8))                     \
    VUNPCKLPS(ZMM(  6), ZMM( Z9), ZMM(Z10))                     \
    VUNPCKHPS(ZMM(Z10), ZMM( Z9), ZMM(Z10))                     \
    VUNPCKHPS(ZMM(  7), ZMM(Z11), ZMM(Z12))                     \
    VUNPCKLPS(ZMM(Z11), ZMM(Z11), ZMM(Z12))                     \
    /* 00 10 02 12 04 14 06 16 08 18 0A 1A 0C 1C 0E 1E -   2 */ \
    /* 01 11 03 13 05 15 07 17 09 19 0B 1B 0D 1D 0F 1F -  Z2 */ \
    /* 20 30 22 32 24 34 26 36 28 38 2A 3A 2C 3C 2E 3E -   3 */ \
    /* 21 31 23 33 25 35 27 37 29 39 2B 3B 2D 3D 2F 3F -  Z4 */ \
    /* 40 50 42 52 44 54 46 56 48 58 4A 5A 4C 5C 4E 5E -   4 */ \
    /* 41 51 43 53 45 55 47 57 49 59 4B 5B 4D 5D 4F 5F -  Z6 */ \
    /* 60 70 62 72 64 74 66 76 68 78 6A 7A 6C 7C 6E 7E -   5 */ \
    /* 61 71 63 73 65 75 67 77 69 79 6B 7B 6D 7D 6F 7F -  Z8 */ \
    /* 80 90 82 92 84 94 86 96 88 98 8A 8A 8C 9C 8E 9E -   6 */ \
    /* 81 91 83 93 85 95 87 97 89 99 8B 9B 8D 9D 8F 9F - Z10 */ \
    /* A0 B0 A2 B2 A4 B4 A6 B6 A8 B8 AA BA AC BC AE BE - Z11 */ \
    /* A1 B1 A3 B3 A5 B5 A7 B7 A9 B9 AB BB AD BD AF BF -   7 */ \
    VSHUFPS(ZMM( Z3), ZMM(  2), ZMM(  3), IMM(0xee))            \
    VSHUFPS(ZMM(  2), ZMM(  2), ZMM(  3), IMM(0x44))            \
    VSHUFPS(ZMM(  3), ZMM( Z2), ZMM( Z4), IMM(0xee))            \
    VSHUFPS(ZMM( Z2), ZMM( Z2), ZMM( Z4), IMM(0x44))            \
    VSHUFPS(ZMM( Z5), ZMM(  4), ZMM(  5), IMM(0x44))            \
    VSHUFPS(ZMM(  5), ZMM(  4), ZMM(  5), IMM(0xee))            \
    VSHUFPS(ZMM(  4), ZMM( Z6), ZMM( Z8), IMM(0x44))            \
    VSHUFPS(ZMM( Z8), ZMM( Z6), ZMM( Z8), IMM(0xee))            \
    VSHUFPS(ZMM( Z9), ZMM(  6), ZMM(Z11), IMM(0x44))            \
    VSHUFPS(ZMM(Z11), ZMM(  6), ZMM(Z11), IMM(0xee))            \
    VSHUFPS(ZMM(Z12), ZMM(Z10), ZMM(  7), IMM(0xee))            \
    VSHUFPS(ZMM(Z10), ZMM(Z10), ZMM(  7), IMM(0x44))            \
    /* 00 10 20 30 04 14 24 34 08 18 28 38 0C 1C 2C 3C -   2 */ \
    /* 01 11 03 31 05 15 25 35 09 19 29 39 0D 1D 2D 3D -  Z2 */ \
    /* 02 12 22 32 06 16 26 36 0A 1A 2A 3A 0E 1E 2E 3E -  Z3 */ \
    /* 03 13 23 33 07 17 27 37 0B 1B 2B 3B 0F 1F 2F 3F -   3 */ \
    /* 40 50 60 70 44 54 64 74 48 58 68 78 4C 5C 6C 7C -  Z5 */ \
    /* 41 51 61 71 45 55 65 75 49 59 69 79 4D 5D 6D 7D -   4 */ \
    /* 42 52 62 72 46 56 66 76 4A 5A 6A 7A 4E 5E 6E 7E -   5 */ \
    /* 43 53 63 73 47 57 67 77 4B 5B 6B 7B 4F 5F 6F 7F -  Z8 */ \
    /* 80 90 A0 B0 84 94 A4 B4 88 98 A8 B8 8C 9C AC BC -  Z9 */ \
    /* 81 91 A1 B1 85 95 A5 B5 89 99 A9 B9 8D 9D AD BD - Z10 */ \
    /* 82 92 A2 B2 86 96 A6 B6 8A 9A AA BA 8E 9E AE BE - Z11 */ \
    /* 83 93 A3 B3 87 97 A7 B7 8B 9B AB BB 8F 9F AF BF - Z12 */ \
    VSHUFF32X4(ZMM( Z1), ZMM(  2), ZMM( Z5), IMM(0x44))         \
    VSHUFF32X4(ZMM( Z5), ZMM(  2), ZMM( Z5), IMM(0xee))         \
    VSHUFF32X4(ZMM( Z6), ZMM( Z2), ZMM(  4), IMM(0xee))         \
    VSHUFF32X4(ZMM( Z2), ZMM( Z2), ZMM(  4), IMM(0x44))         \
    VSHUFF32X4(ZMM( Z7), ZMM( Z3), ZMM(  5), IMM(0xee))         \
    VSHUFF32X4(ZMM( Z3), ZMM( Z3), ZMM(  5), IMM(0x44))         \
    VSHUFF32X4(ZMM( Z4), ZMM(  3), ZMM( Z8), IMM(0x44))         \
    VSHUFF32X4(ZMM( Z8), ZMM(  3), ZMM( Z8), IMM(0xee))         \
    /* 00 10 20 30 04 14 24 34 40 50 60 70 44 54 64 74 -  Z1 */ \
    /* 01 11 03 31 05 15 25 35 41 51 61 71 45 55 65 75 -  Z2 */ \
    /* 02 12 22 32 06 16 26 36 42 52 62 72 46 56 66 76 -  Z3 */ \
    /* 03 13 23 33 07 17 27 37 43 53 63 73 47 57 67 77 -  Z4 */ \
    /* 08 18 28 38 0C 1C 2C 3C 48 58 68 78 4C 5C 6C 7C -  Z5 */ \
    /* 09 19 29 39 0D 1D 2D 3D 49 59 69 79 4D 5D 6D 7D -  Z6 */ \
    /* 0A 1A 2A 3A 0E 1E 2E 3E 4A 5A 6A 7A 4E 5E 6E 7E -  Z7 */ \
    /* 0B 1B 2B 3B 0F 1F 2F 3F 4B 5B 6B 7B 4F 5F 6F 7F -  Z8 */ \
    /* 80 90 A0 B0 84 94 A4 B4 88 98 A8 B8 8C 9C AC BC -  Z9 */ \
    /* 81 91 A1 B1 85 95 A5 B5 89 99 A9 B9 8D 9D AD BD - Z10 */ \
    /* 82 92 A2 B2 86 96 A6 B6 8A 9A AA BA 8E 9E AE BE - Z11 */ \
    /* 83 93 A3 B3 87 97 A7 B7 8B 9B AB BB 8F 9F AF BF - Z12 */ \
    VSHUFF32X4(ZMM(  2), ZMM(Z1), ZMM( Z9), IMM(0x08))          \
    VSHUFF32X4(ZMM(  3), ZMM(Z2), ZMM(Z10), IMM(0x08))          \
    VSHUFF32X4(ZMM(  4), ZMM(Z3), ZMM(Z11), IMM(0x08))          \
    VSHUFF32X4(ZMM(  5), ZMM(Z4), ZMM(Z12), IMM(0x5d))          \
    VSHUFF32X4(ZMM(  6), ZMM(Z1), ZMM( Z9), IMM(0x5d))          \
    VSHUFF32X4(ZMM(  7), ZMM(Z2), ZMM(Z10), IMM(0x5d))          \
    VSHUFF32X4(ZMM( Z1), ZMM(Z3), ZMM(Z11), IMM(0x5d))          \
    VSHUFF32X4(ZMM( Z2), ZMM(Z4), ZMM(Z12), IMM(0x5d))          \
    VSHUFF32X4(ZMM( Z3), ZMM(Z5), ZMM( Z9), IMM(0xa8))          \
    VSHUFF32X4(ZMM( Z4), ZMM(Z6), ZMM(Z10), IMM(0xa8))          \
    VSHUFF32X4(ZMM( Z7), ZMM(Z7), ZMM(Z11), IMM(0xa8))          \
    VSHUFF32X4(ZMM( Z8), ZMM(Z8), ZMM(Z12), IMM(0xa8))          \
    VSHUFF32X4(ZMM( Z9), ZMM(Z5), ZMM( Z9), IMM(0xfd))          \
    VSHUFF32X4(ZMM(Z10), ZMM(Z6), ZMM(Z10), IMM(0xfd))          \
    VSHUFF32X4(ZMM(Z11), ZMM(Z7), ZMM(Z11), IMM(0xfd))          \
    VSHUFF32X4(ZMM(Z12), ZMM(Z8), ZMM(Z12), IMM(0xfd))          \
    /* 00 10 20 30 40 50 60 70 80 90 A0 B0 80 90 A0 B0 -   2 */ \
    /* 01 11 03 31 41 51 61 71 81 91 A1 B1 81 91 A1 B1 -   3 */ \
    /* 02 12 22 32 42 52 62 72 82 92 A2 B2 82 92 A2 B2 -   4 */ \
    /* 03 13 23 33 43 53 63 73 83 93 A3 B3 83 93 A3 B3 -   5 */ \
    /* 04 14 24 34 44 54 64 74 84 94 A4 B4 84 94 A4 B4 -   6 */ \
    /* 05 15 25 35 45 55 65 75 85 95 A5 B5 85 95 A5 B5 -   7 */ \
    /* 06 16 26 36 46 56 66 76 86 96 A6 B6 86 96 A6 B6 -  Z1 */ \
    /* 07 17 27 37 47 57 67 77 87 97 A7 B7 87 97 A7 B7 -  Z2 */ \
    /* 08 18 28 38 48 58 68 78 88 98 A8 B8 88 98 A8 B8 -  Z3 */ \
    /* 09 19 09 39 49 59 69 79 89 99 A9 B9 89 99 A9 B9 -  Z4 */ \
    /* 0A 1A 2A 3A 4A 5A 6A 7A 8A 9A AA BA 8A 9A AA BA -  Z7 */ \
    /* 0B 1B 2B 3B 4B 5B 6B 7B 8B 9B AB BB 8B 9B AB BB -  Z8 */ \
    /* 0C 1C 2C 3C 4C 5C 6C 7C 8C 9C AC BC 8C 9C AC BC -  Z9 */ \
    /* 0D 1D 2D 3D 4D 5D 6D 7D 8D 9D AD BD 8D 9D AD BD - Z10 */ \
    /* 0E 1E 2E 3E 4E 5E 6E 7E 8E 9E AE BE 8E 9E AE BE - Z11 */ \
    /* 0F 1F 2F 3F 4F 5F 6F 7F 8F 9F AF BF 8F 9F AF BF - Z12 */ \
    VMOVUPS(MEM(RCX      ) MASK_K(1), ZMM(  2))                 \
    VMOVUPS(MEM(RCX,RBX,1) MASK_K(1), ZMM(  3))                 \
    VMOVUPS(MEM(RCX,RBX,2) MASK_K(1), ZMM(  4))                 \
    VMOVUPS(MEM(RCX,R13,1) MASK_K(1), ZMM(  5))                 \
    VMOVUPS(MEM(RCX,RBX,4) MASK_K(1), ZMM(  6))                 \
    VMOVUPS(MEM(RCX,R15,1) MASK_K(1), ZMM(  7))                 \
    VMOVUPS(MEM(RCX,R13,2) MASK_K(1), ZMM( Z1))                 \
    VMOVUPS(MEM(RCX,R10,1) MASK_K(1), ZMM( Z2))                 \
    VMOVUPS(MEM(RDX      ) MASK_K(1), ZMM( Z3))                 \
    VMOVUPS(MEM(RDX,RBX,1) MASK_K(1), ZMM( Z4))                 \
    VMOVUPS(MEM(RDX,RBX,2) MASK_K(1), ZMM( Z7))                 \
    VMOVUPS(MEM(RDX,R13,1) MASK_K(1), ZMM( Z8))                 \
    VMOVUPS(MEM(RDX,RBX,4) MASK_K(1), ZMM( Z9))                 \
    VMOVUPS(MEM(RDX,R15,1) MASK_K(1), ZMM(Z10))                 \
    VMOVUPS(MEM(RDX,R13,2) MASK_K(1), ZMM(Z11))                 \
    VMOVUPS(MEM(RDX,R10,1) MASK_K(1), ZMM(Z12))

#ifdef PREFETCH_C_L2
#undef PREFETCH_C_L2
#define PREFETCH_C_L2 \
\
    PREFETCH(1, MEM(RCX,      0*64)) \
    PREFETCH(1, MEM(RCX,      1*64)) \
    \
    PREFETCH(1, MEM(RCX,R12,1,0*64)) \
    PREFETCH(1, MEM(RCX,R12,1,1*64)) \
    \
    PREFETCH(1, MEM(RCX,R12,2,0*64)) \
    PREFETCH(1, MEM(RCX,R12,2,1*64)) \
    \
    PREFETCH(1, MEM(RCX,R13,1,0*64)) \
    PREFETCH(1, MEM(RCX,R13,1,1*64)) \
    \
    PREFETCH(1, MEM(RCX,R12,4,0*64)) \
    PREFETCH(1, MEM(RCX,R12,4,1*64)) \
    \
    PREFETCH(1, MEM(RCX,R14,1,0*64)) \
    PREFETCH(1, MEM(RCX,R14,1,1*64)) \
    \
    PREFETCH(1, MEM(RCX,R13,2,0*64)) \
    PREFETCH(1, MEM(RCX,R13,2,1*64)) \
    \
    PREFETCH(1, MEM(RCX,R15,1,0*64)) \
    PREFETCH(1, MEM(RCX,R15,1,1*64)) \
    \
    PREFETCH(1, MEM(RDX,      0*64)) \
    PREFETCH(1, MEM(RDX,      1*64)) \
    \
    PREFETCH(1, MEM(RDX,R12,1,0*64)) \
    PREFETCH(1, MEM(RDX,R12,1,1*64)) \
    \
    PREFETCH(1, MEM(RDX,R12,2,0*64)) \
    PREFETCH(1, MEM(RDX,R12,2,1*64)) \
    \
    PREFETCH(1, MEM(RDX,R13,1,0*64)) \
    PREFETCH(1, MEM(RDX,R13,1,1*64))

#else
#undef PREFETCH_C_L2
#define PREFETCH_C_L2
#endif


#define PREFETCH_C_L1 \
\
    PREFETCHW0(MEM(RCX,      0*64)) \
    PREFETCHW0(MEM(RCX,      1*64)) \
    PREFETCHW0(MEM(RCX,R12,1,0*64)) \
    PREFETCHW0(MEM(RCX,R12,1,1*64)) \
    PREFETCHW0(MEM(RCX,R12,2,0*64)) \
    PREFETCHW0(MEM(RCX,R12,2,1*64)) \
    PREFETCHW0(MEM(RCX,R13,1,0*64)) \
    PREFETCHW0(MEM(RCX,R13,1,1*64)) \
    PREFETCHW0(MEM(RCX,R12,4,0*64)) \
    PREFETCHW0(MEM(RCX,R12,4,1*64)) \
    PREFETCHW0(MEM(RCX,R14,1,0*64)) \
    PREFETCHW0(MEM(RCX,R14,1,1*64)) \
    PREFETCHW0(MEM(RCX,R13,2,0*64)) \
    PREFETCHW0(MEM(RCX,R13,2,1*64)) \
    PREFETCHW0(MEM(RCX,R15,1,0*64)) \
    PREFETCHW0(MEM(RCX,R15,1,1*64)) \
    PREFETCHW0(MEM(RDX,      0*64)) \
    PREFETCHW0(MEM(RDX,      1*64)) \
    PREFETCHW0(MEM(RDX,R12,1,0*64)) \
    PREFETCHW0(MEM(RDX,R12,1,1*64)) \
    PREFETCHW0(MEM(RDX,R12,2,0*64)) \
    PREFETCHW0(MEM(RDX,R12,2,1*64)) \
    PREFETCHW0(MEM(RDX,R13,1,0*64)) \
    PREFETCHW0(MEM(RDX,R13,1,1*64))

//
// n: index in unrolled loop
//
// a: ZMM register to load into
// b: ZMM register to read from
//
// ...: addressing for A, except for offset
//
#define SUBITER(n) \
\
    PREFETCH_B_L1(n, 0) \
    \
    VBROADCASTSS(ZMM(3), MEM(RAX,(12*n+ 0)*4)) \
    VBROADCASTSS(ZMM(4), MEM(RAX,(12*n+ 1)*4)) \
    VFMADD231PS(ZMM( 8), ZMM(0), ZMM(3)) \
    VFMADD231PS(ZMM( 9), ZMM(1), ZMM(3)) \
    VFMADD231PS(ZMM(10), ZMM(0), ZMM(4)) \
    VFMADD231PS(ZMM(11), ZMM(1), ZMM(4)) \
    \
    VBROADCASTSS(ZMM(3), MEM(RAX,(12*n+ 2)*4)) \
    VBROADCASTSS(ZMM(4), MEM(RAX,(12*n+ 3)*4)) \
    VFMADD231PS(ZMM(12), ZMM(0), ZMM(3)) \
    VFMADD231PS(ZMM(13), ZMM(1), ZMM(3)) \
    VFMADD231PS(ZMM(14), ZMM(0), ZMM(4)) \
    VFMADD231PS(ZMM(15), ZMM(1), ZMM(4)) \
    \
    VBROADCASTSS(ZMM(3), MEM(RAX,(12*n+ 4)*4)) \
    VBROADCASTSS(ZMM(4), MEM(RAX,(12*n+ 5)*4)) \
    VFMADD231PS(ZMM(16), ZMM(0), ZMM(3)) \
    VFMADD231PS(ZMM(17), ZMM(1), ZMM(3)) \
    VFMADD231PS(ZMM(18), ZMM(0), ZMM(4)) \
    VFMADD231PS(ZMM(19), ZMM(1), ZMM(4)) \
    \
    PREFETCH_B_L1(n, 1) \
    \
    VBROADCASTSS(ZMM(3), MEM(RAX,(12*n+ 6)*4)) \
    VBROADCASTSS(ZMM(4), MEM(RAX,(12*n+ 7)*4)) \
    VFMADD231PS(ZMM(20), ZMM(0), ZMM(3)) \
    VFMADD231PS(ZMM(21), ZMM(1), ZMM(3)) \
    VFMADD231PS(ZMM(22), ZMM(0), ZMM(4)) \
    VFMADD231PS(ZMM(23), ZMM(1), ZMM(4)) \
    \
    VBROADCASTSS(ZMM(3), MEM(RAX,(12*n+ 8)*4)) \
    VBROADCASTSS(ZMM(4), MEM(RAX,(12*n+ 9)*4)) \
    VFMADD231PS(ZMM(24), ZMM(0), ZMM(3)) \
    VFMADD231PS(ZMM(25), ZMM(1), ZMM(3)) \
    VFMADD231PS(ZMM(26), ZMM(0), ZMM(4)) \
    VFMADD231PS(ZMM(27), ZMM(1), ZMM(4)) \
    \
    VBROADCASTSS(ZMM(3), MEM(RAX,(12*n+10)*4)) \
    VBROADCASTSS(ZMM(4), MEM(RAX,(12*n+11)*4)) \
    VFMADD231PS(ZMM(28), ZMM(0), ZMM(3)) \
    VFMADD231PS(ZMM(29), ZMM(1), ZMM(3)) \
    VFMADD231PS(ZMM(30), ZMM(0), ZMM(4)) \
    VFMADD231PS(ZMM(31), ZMM(1), ZMM(4)) \
    \
    VMOVAPS(ZMM(0), MEM(RBX,(32*n+ 0)*4)) \
    VMOVAPS(ZMM(1), MEM(RBX,(32*n+16)*4))

//This is an array used for the scatter/gather instructions.
static int64_t offsets[16] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15};

void bli_sgemm_opt_12x32_l2(
                             dim_t            k_,
                             float* restrict alpha,
                             float* restrict a,
                             float* restrict b,
                             float* restrict beta,
                             float* restrict c, inc_t rs_c_, inc_t cs_c_,
                             auxinfo_t*       data,
                             cntx_t* restrict cntx
                           )
{
    (void)data;
    (void)cntx;

    const int64_t* offsetPtr = &offsets[0];
    const int64_t k = k_;
    const int64_t rs_c = rs_c_;
    const int64_t cs_c = cs_c_;

    __asm__ volatile
    (

    VXORPD(YMM(8), YMM(8), YMM(8)) //clear out registers
    VMOVAPD(YMM( 7), YMM(8))
    VMOVAPD(YMM( 9), YMM(8))
    VMOVAPD(YMM(10), YMM(8))   MOV(RSI, VAR(k)) //loop index
    VMOVAPD(YMM(11), YMM(8))   MOV(RAX, VAR(a)) //load address of a
    VMOVAPD(YMM(12), YMM(8))   MOV(RBX, VAR(b)) //load address of b
    VMOVAPD(YMM(13), YMM(8))   MOV(RCX, VAR(c)) //load address of c
    VMOVAPD(YMM(14), YMM(8))
    VMOVAPD(YMM(15), YMM(8))   VMOVAPS(ZMM(0), MEM(RBX,  0*4)) //pre-load b
    VMOVAPD(YMM(16), YMM(8))   VMOVAPS(ZMM(1), MEM(RBX, 16*4)) //pre-load b
    VMOVAPD(YMM(17), YMM(8))
    VMOVAPD(YMM(18), YMM(8))
    VMOVAPD(YMM(19), YMM(8))   MOV(R12, VAR(rs_c))      //rs_c
    VMOVAPD(YMM(20), YMM(8))   LEA(R13, MEM(R12,R12,2)) //*3
    VMOVAPD(YMM(21), YMM(8))   LEA(R14, MEM(R12,R12,4)) //*5
    VMOVAPD(YMM(22), YMM(8))   LEA(R15, MEM(R14,R12,2)) //*7
    VMOVAPD(YMM(23), YMM(8))   LEA(RDX, MEM(RCX,R12,8)) //c + 8*rs_c
    VMOVAPD(YMM(24), YMM(8))
    VMOVAPD(YMM(25), YMM(8))   MOV(R8, IMM(12*4)) //mr*sizeof(float)
    VMOVAPD(YMM(26), YMM(8))   MOV(R9, IMM(32*4)) //nr*sizeof(float)
    VMOVAPD(YMM(27), YMM(8))
    VMOVAPD(YMM(28), YMM(8))   LEA(RBX, MEM(RBX,R9,1)) //adjust b for pre-load
    VMOVAPD(YMM(29), YMM(8))
    VMOVAPD(YMM(30), YMM(8))
    VMOVAPD(YMM(31), YMM(8))

    TEST(RSI, RSI)
    JZ(POSTACCUM)

#ifdef PREFETCH_A_BEFORE
    PREFETCH(0, MEM(RAX,0*64))
    PREFETCH(0, MEM(RAX,1*64))
    PREFETCH(0, MEM(RAX,2*64))
#endif

#ifdef PREFETCH_B_BEFORE
    PREFETCH(0, MEM(RBX,0*64))
    PREFETCH(0, MEM(RBX,1*64))
    PREFETCH(0, MEM(RBX,2*64))
    PREFETCH(0, MEM(RBX,3*64))
    PREFETCH(0, MEM(RBX,4*64))
    PREFETCH(0, MEM(RBX,5*64))
    PREFETCH(0, MEM(RBX,6*64))
    PREFETCH(0, MEM(RBX,7*64))
#endif

    PREFETCH_C_L2

    MOV(RDI, RSI)
    AND(RSI, IMM(3))
    SAR(RDI, IMM(2))

    SUB(RDI, IMM(0+TAIL_NITER))
    JLE(K_SMALL)

    LOOP_ALIGN
    LABEL(MAIN_LOOP)

        PREFETCH(0, MEM(RAX,A_L1_PREFETCH_DIST*12*4))
        SUBITER(0)
        PREFETCH(0, MEM(RAX,A_L1_PREFETCH_DIST*12*4+64))
        SUBITER(1)
        PREFETCH(0, MEM(RAX,A_L1_PREFETCH_DIST*12*4+128))
        SUBITER(2)
        SUBITER(3)

        LEA(RAX, MEM(RAX,R8,4))
        LEA(RBX, MEM(RBX,R9,4))

        DEC(RDI)

    JNZ(MAIN_LOOP)

    LABEL(K_SMALL)

    PREFETCH_C_L1

    ADD(RDI, IMM(0+TAIL_NITER))
    JZ(TAIL_LOOP)

    LOOP_ALIGN
    LABEL(SMALL_LOOP)

        PREFETCH(0, MEM(RAX,A_L1_PREFETCH_DIST*12*4))
        SUBITER(0)
        PREFETCH(0, MEM(RAX,A_L1_PREFETCH_DIST*12*4+64))
        SUBITER(1)
        PREFETCH(0, MEM(RAX,A_L1_PREFETCH_DIST*12*4+128))
        SUBITER(2)
        SUBITER(3)

        LEA(RAX, MEM(RAX,R8,4))
        LEA(RBX, MEM(RBX,R9,4))

        DEC(RDI)

    JNZ(SMALL_LOOP)

    TEST(RSI, RSI)
    JZ(POSTACCUM)

    LOOP_ALIGN
    LABEL(TAIL_LOOP)

        PREFETCH(0, MEM(RAX,A_L1_PREFETCH_DIST*12*4))
        SUBITER(0)

        ADD(RAX, R8)
        ADD(RBX, R9)

        DEC(RSI)

    JNZ(TAIL_LOOP)

    LABEL(POSTACCUM)

#ifdef PREFETCH_A_AFTER
    MOV(R8, VAR(a))
    PREFETCH(0, MEM(R8,0*64))
    PREFETCH(0, MEM(R8,1*64))
    PREFETCH(0, MEM(R8,2*64))
#endif

#ifdef PREFETCH_B_AFTER
    MOV(R9, VAR(b))
    PREFETCH(0, MEM(R9,0*64))
    PREFETCH(0, MEM(R9,1*64))
    PREFETCH(0, MEM(R9,2*64))
    PREFETCH(0, MEM(R9,3*64))
    PREFETCH(0, MEM(R9,4*64))
    PREFETCH(0, MEM(R9,5*64))
    PREFETCH(0, MEM(R9,6*64))
    PREFETCH(0, MEM(R9,7*64))
#endif

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSS(ZMM(0), MEM(RAX))
    VBROADCASTSS(ZMM(1), MEM(RBX))

    MOV(RAX, VAR(rs_c))
    LEA(RAX, MEM(,RAX,4))
    MOV(RBX, VAR(cs_c))
    LEA(RBX, MEM(,RBX,4))

    // Check if C is row stride.
    CMP(RBX, IMM(4))
    JNE(COLSTORED)

        VCOMISS(XMM(1), XMM(7))
        JE(ROWSTORBZ)

            UPDATE_C( 8, 9,10,11)
            UPDATE_C(12,13,14,15)
            UPDATE_C(16,17,18,19)
            UPDATE_C(20,21,22,23)
            UPDATE_C(24,25,26,27)
            UPDATE_C(28,29,30,31)

        JMP(END)
        LABEL(ROWSTORBZ)

            UPDATE_C_BZ( 8, 9,10,11)
            UPDATE_C_BZ(12,13,14,15)
            UPDATE_C_BZ(16,17,18,19)
            UPDATE_C_BZ(20,21,22,23)
            UPDATE_C_BZ(24,25,26,27)
            UPDATE_C_BZ(28,29,30,31)

    JMP(END)
    LABEL(COLSTORED)

    // Check if C is column stride. If not, jump to the slow scattered update
    CMP(RAX, IMM(4))
    JNE(SCATTEREDUPDATE)

        //
        // Transpose and write out in two halves, only low half shown:
        //
        // +---------------------------------------------------------------+
        // |c00-c01-c02-c03-c04-c05-c06-c07-c08-c09-c0A-c0B-c0C-c0D-c0E-c0F|
        // |                                                               |
        // |c10-c11-c12-c13-c14-c15-c16-c17-c18-c19-c1A-c1B-c1C-c1D-c1E-c1F|
        // |                                                               |
        // |c20-c21-c22-c23-c24-c25-c26-c27-c28-c29-c2A-c2B-c2C-c2D-c2E-c2F|
        // |                                                               |
        // |c30-c31-c32-c33-c34-c35-c36-c37-c38-c39-c3A-c3B-c3C-c3D-c3E-c3F|
        // |                                                               |
        // |c40-c41-c42-c43-c44-c45-c46-c47-c48-c49-c4A-c4B-c4C-c4D-c4E-c4F|
        // |                                                               |
        // |c50-c51-c52-c53-c54-c55-c56-c57-c58-c59-c5A-c5B-c5C-c5D-c5E-c5F|
        // |                                                               |
        // |c60-c61-c62-c63-c64-c65-c66-c67-c68-c69-c6A-c6B-c6C-c6D-c6E-c6F|
        // |                                                               |
        // |c70-c71-c72-c73-c74-c75-c76-c77-c78-c79-c7A-c7B-c7C-c7D-c7E-c7F|
        // |                                                               |
        // |c80-c81-c82-c83-c84-c85-c86-c87-c88-c89-c8A-c8B-c8C-c8D-c8E-c8F|
        // |                                                               |
        // |c90-c91-c92-c93-c94-c95-c96-c97-c98-c99-c9A-c9B-c9C-c9D-c9E-c9F|
        // |                                                               |
        // |cA0-cA1-cA2-cA3-cA4-cA5-cA6-cA7-cA8-cA9-cAA-cAB-cAC-cAD-cAE-cAF|
        // |                                                               |
        // |cB0-cB1-cB2-cB3-cB4-cB5-cB6-cB7-cB8-cB9-cBA cBB-cBC-cBD-cBE-cBF|
        // +---------------------------------------------------------------+
        //
        //                                ||
        //                                \/
        //
        // +---------------------------------------------------------------+
        // |c00 c01 c02 c03 c04 c05 c06 c07 c08 c09 c0A c0B c0C c0D c0E c0F|
        // | |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | |
        // |c10 c11 c12 c13 c14 c15 c16 c17 c18 c19 c1A c1B c1C c1D c1E c1F|
        // | |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | |
        // |c20 c21 c22 c23 c24 c25 c26 c27 c28 c29 c2A c2B c2C c2D c2E c2F|
        // | |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | |
        // |c30 c31 c32 c33 c34 c35 c36 c37 c38 c39 c3A c3B c3C c3D c3E c3F|
        // | |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | |
        // |c40 c41 c42 c43 c44 c45 c46 c47 c48 c49 c4A c4B c4C c4D c4E c4F|
        // | |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | |
        // |c50 c51 c52 c53 c54 c55 c56 c57 c58 c59 c5A c5B c5C c5D c5E c5F|
        // | |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | |
        // |c60 c61 c62 c63 c64 c65 c66 c67 c68 c69 c6A c6B c6C c6D c6E c6F|
        // | |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | |
        // |c70 c71 c72 c73 c74 c75 c76 c77 c78 c79 c7A c7B c7C c7D c7E c7F|
        // | |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | |
        // |c80 c81 c82 c83 c84 c85 c86 c87 c88 c89 c8A c8B c8C c8D c8E c8F|
        // | |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | |
        // |c90 c91 c92 c93 c94 c95 c96 c97 c98 c99 c9A c9B c9C c9D c9E c9F|
        // | |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | |
        // |cA0 cA1 cA2 cA3 cA4 cA5 cA6 cA7 cA8 cA9 cAA cAB cAC cAD cAE cAF|
        // | |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | |
        // |cB0 cB1 cB2 cB3 cB4 cB5 cB6 cB7 cB8 cB9 cBA cBB cBC cBD cBE cBF|
        // +---------------------------------------------------------------+
        //

        MOV(ESI, IMM(0xfff))
        KMOV(K(1), ESI)

        LEA(R13, MEM(RBX,RBX,2))
        LEA(R15, MEM(RBX,RBX,4))
        LEA(R10, MEM(R15,RBX,2))
        LEA( R8, MEM(   ,RBX,8))
        LEA(RDX, MEM(RCX,RBX,8))

        VCOMISD(XMM(1), XMM(7))
        JE(COLSTORBZ)

            UPDATE_C_TRANS16X12(8,10,12,14,16,18,20,22,24,26,28,30)

            LEA(RCX, MEM(RCX,R8,2))
            LEA(RDX, MEM(RDX,R8,2))

            UPDATE_C_TRANS16X12(8,10,12,14,16,18,20,22,24,26,28,30)

        JMP(END)
        LABEL(COLSTORBZ)

            UPDATE_C_TRANS16X12_BZ(8,10,12,14,16,18,20,22,24,26,28,30)

            LEA(RCX, MEM(RCX,R8,2))
            LEA(RDX, MEM(RDX,R8,2))

            UPDATE_C_TRANS16X12_BZ(8,10,12,14,16,18,20,22,24,26,28,30)

    JMP(END)
    LABEL(SCATTEREDUPDATE)

        LEA(RDX, MEM(RCX,RBX,8))
        LEA(RDX, MEM(RDX,RBX,8))

        MOV(RDI, VAR(offsetPtr))
        VMOVDQA64(ZMM(2), MEM(RDI,0*64))
        VMOVDQA64(ZMM(3), MEM(RDI,1*64))
        VPBROADCASTQ(ZMM(6), RBX)
        VPMULLQ(ZMM(2), ZMM(6), ZMM(2))
        VPMULLQ(ZMM(3), ZMM(6), ZMM(3))

        VCOMISS(XMM(1), XMM(7))
        JE(SCATTERBZ)

            UPDATE_C_ROW_SCATTERED( 8, 9,10,11)
            UPDATE_C_ROW_SCATTERED(12,13,14,15)
            UPDATE_C_ROW_SCATTERED(16,17,18,19)
            UPDATE_C_ROW_SCATTERED(20,21,22,23)
            UPDATE_C_ROW_SCATTERED(24,25,26,27)
            UPDATE_C_ROW_SCATTERED(28,29,30,31)

        JMP(END)
        LABEL(SCATTERBZ)

            UPDATE_C_BZ_ROW_SCATTERED( 8, 9,10,11)
            UPDATE_C_BZ_ROW_SCATTERED(12,13,14,15)
            UPDATE_C_BZ_ROW_SCATTERED(16,17,18,19)
            UPDATE_C_BZ_ROW_SCATTERED(20,21,22,23)
            UPDATE_C_BZ_ROW_SCATTERED(24,25,26,27)
            UPDATE_C_BZ_ROW_SCATTERED(28,29,30,31)

    LABEL(END)

    VZEROUPPER()

    : // output operands
    : // input operands
      [k]         "m" (k),
      [a]         "m" (a),
      [b]         "m" (b),
      [alpha]     "m" (alpha),
      [beta]      "m" (beta),
      [c]         "m" (c),
      [rs_c]      "m" (rs_c),
      [cs_c]      "m" (cs_c),
      [offsetPtr] "m" (offsetPtr)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12",
      "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
      "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
      "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
      "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
      "zmm30", "zmm31", "memory"
    );
}
