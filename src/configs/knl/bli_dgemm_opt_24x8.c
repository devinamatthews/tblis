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

#define UNROLL_K 32

#define A_L1_PREFETCH_DIST 18
#define B_L1_PREFETCH_DIST 18

#define LOOP_ALIGN ALIGN16

#define UPDATE_C_FOUR_ROWS(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VFMADD231PD(ZMM(R1), ZMM(1), MEM(RCX      )) \
    VFMADD231PD(ZMM(R2), ZMM(1), MEM(RCX,RAX,1)) \
    VFMADD231PD(ZMM(R3), ZMM(1), MEM(RCX,RAX,2)) \
    VFMADD231PD(ZMM(R4), ZMM(1), MEM(RCX,RDI,1)) \
    VMOVUPD(MEM(RCX      ), ZMM(R1)) \
    VMOVUPD(MEM(RCX,RAX,1), ZMM(R2)) \
    VMOVUPD(MEM(RCX,RAX,2), ZMM(R3)) \
    VMOVUPD(MEM(RCX,RDI,1), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,4))

#define UPDATE_C_BZ_FOUR_ROWS(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VMOVUPD(MEM(RCX      ), ZMM(R1)) \
    VMOVUPD(MEM(RCX,RAX,1), ZMM(R2)) \
    VMOVUPD(MEM(RCX,RAX,2), ZMM(R3)) \
    VMOVUPD(MEM(RCX,RDI,1), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,4))

#define UPDATE_C_ROW_SCATTERED(NUM) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPD(ZMM(NUM), ZMM(NUM), ZMM(0)) \
    VGATHERDPD(ZMM(3) MASK_K(1), MEM(RCX,YMM(2),1)) \
    VFMADD231PD(ZMM(NUM), ZMM(3), ZMM(1)) \
    VSCATTERDPD(MEM(RCX,YMM(2),1) MASK_K(2), ZMM(NUM)) \
    ADD(RCX, RAX)

#define UPDATE_C_BZ_ROW_SCATTERED(NUM) \
\
    KXNORW(K(1), K(0), K(0)) \
    VMULPD(ZMM(NUM), ZMM(NUM), ZMM(0)) \
    VSCATTERDPD(MEM(RCX,YMM(2),1) MASK_K(1), ZMM(NUM)) \
    ADD(RCX, RAX)

//
// zmm{r1-r8}: eight rows of AB
// zmm0:       alpha
// zmm1:       beta
// zmm2-7:     temporaries, will be overwritten
// k1:         0x33
// rcx:        C
// rbx:        rs_c
// r13:        rs_c*3
// r15:        rs_c*5
// r10:        rs_c*7
#define UPDATE_C_TRANS8X8(R1,R2,R3,R4,R5,R6,R7,R8) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VMULPD(ZMM(R5), ZMM(R5), ZMM(0)) \
    VMULPD(ZMM(R6), ZMM(R6), ZMM(0)) \
    VMULPD(ZMM(R7), ZMM(R7), ZMM(0)) \
    VMULPD(ZMM(R8), ZMM(R8), ZMM(0)) \
    /* 00 01 02 03 04 05 06 07 - R1 */ \
    /* 10 11 12 13 14 15 16 17 - R2 */ \
    /* 20 21 22 23 24 25 26 27 - R3 */ \
    /* 30 31 32 33 34 35 36 37 - R4 */ \
    /* 40 41 42 43 44 45 46 47 - R5 */ \
    /* 50 51 52 53 54 55 56 57 - R6 */ \
    /* 60 61 62 63 64 65 66 67 - R7 */ \
    /* 70 71 72 73 74 75 76 77 - R8 */ \
    VUNPCKLPD(ZMM( 2), ZMM(R1), ZMM(R2)) \
    VUNPCKHPD(ZMM( 3), ZMM(R3), ZMM(R4)) \
    VUNPCKLPD(ZMM( 4), ZMM(R5), ZMM(R6)) \
    VUNPCKHPD(ZMM( 5), ZMM(R7), ZMM(R8)) \
    VUNPCKHPD(ZMM(R2), ZMM(R1), ZMM(R2)) \
    VUNPCKLPD(ZMM(R3), ZMM(R3), ZMM(R4)) \
    VUNPCKHPD(ZMM(R6), ZMM(R5), ZMM(R6)) \
    VUNPCKLPD(ZMM(R7), ZMM(R7), ZMM(R8)) \
    /* 00 10 02 12 04 14 06 16 -  2 */ \
    /* 01 11 03 13 05 15 07 17 - R2 */ \
    /* 20 30 22 32 24 34 26 36 - R3 */ \
    /* 21 31 23 33 25 35 27 37 -  3 */ \
    /* 40 50 42 52 44 54 46 56 -  4 */ \
    /* 41 51 43 53 45 55 47 57 - R6 */ \
    /* 60 70 62 72 64 74 66 76 - R7 */ \
    /* 61 71 63 73 65 75 67 77 -  5 */ \
    VSHUFF64X2(ZMM( 6), ZMM( 2), ZMM(R3), IMM(0x71)) \
    VBLENDMPD(ZMM(R1) MASK_K(1), ZMM( 6), ZMM(R1)) \
    VBLENDMPD(ZMM(R3) MASK_K(1), ZMM(R3), ZMM( 6)) \
    VSHUFF64X2(ZMM( 7), ZMM(R2), ZMM( 3), IMM(0x71)) \
    VBLENDMPD(ZMM(R2) MASK_K(1), ZMM( 7), ZMM(R2)) \
    VBLENDMPD(ZMM(R4) MASK_K(1), ZMM(R4), ZMM( 7)) \
    VSHUFF64X2(ZMM( 2), ZMM( 4), ZMM(R7), IMM(0x71)) \
    VBLENDMPD(ZMM(R3) MASK_K(1), ZMM( 2), ZMM(R3)) \
    VBLENDMPD(ZMM(R5) MASK_K(1), ZMM(R7), ZMM( 2)) \
    VSHUFF64X2(ZMM( 3), ZMM(R6), ZMM( 5), IMM(0x71)) \
    VBLENDMPD(ZMM(R4) MASK_K(1), ZMM( 3), ZMM(R4)) \
    VBLENDMPD(ZMM(R8) MASK_K(1), ZMM(R8), ZMM( 3)) \
    /* 00 10 20 30 02 12 22 32 - R1 */ \
    /* 01 11 21 31 03 13 23 33 - R2 */ \
    /* 04 14 24 34 06 16 26 36 - R3 */ \
    /* 05 15 25 35 07 17 27 37 - R4 */ \
    /* 40 50 60 70 42 52 62 72 - R5 */ \
    /* 41 51 61 71 43 53 63 73 - R6 */ \
    /* 44 54 64 74 46 56 66 76 - R7 */ \
    /* 45 55 65 75 47 57 67 77 - R8 */ \
    VSHUFF64X2(ZMM(R3), ZMM(R1), ZMM( 4), IMM(0xdd)) \
    VSHUFF64X2(ZMM(R1), ZMM(R1), ZMM( 4), IMM(0x88)) \
    VSHUFF64X2(ZMM(R2), ZMM( 2), ZMM(R4), IMM(0x88)) \
    VSHUFF64X2(ZMM(R4), ZMM( 2), ZMM(R4), IMM(0xdd)) \
    VSHUFF64X2(ZMM(R5), ZMM( 3), ZMM(R7), IMM(0x88)) \
    VSHUFF64X2(ZMM(R7), ZMM( 3), ZMM(R7), IMM(0xdd)) \
    VSHUFF64X2(ZMM(R8), ZMM(R6), ZMM( 5), IMM(0xdd)) \
    VSHUFF64X2(ZMM(R6), ZMM(R6), ZMM( 5), IMM(0x88)) \
    /* 00 10 20 30 40 50 60 70 - R1 */ \
    /* 01 11 21 31 41 51 61 71 - R2 */ \
    /* 04 14 24 34 44 54 64 74 - R5 */ \
    /* 05 15 25 35 45 55 65 75 - R6 */ \
    /* 02 12 22 32 42 52 62 72 - R3 */ \
    /* 03 13 23 33 43 53 63 73 - R4 */ \
    /* 06 16 26 36 46 56 66 76 - R7 */ \
    /* 07 17 27 37 47 57 67 77 - R8 */ \
    VFMADD231PD(ZMM(R1), ZMM(1), MEM(RCX      )) \
    VFMADD231PD(ZMM(R2), ZMM(1), MEM(RCX,RBX,1)) \
    VFMADD231PD(ZMM(R3), ZMM(1), MEM(RCX,RBX,2)) \
    VFMADD231PD(ZMM(R4), ZMM(1), MEM(RCX,R13,1)) \
    VFMADD231PD(ZMM(R5), ZMM(1), MEM(RCX,RBX,4)) \
    VFMADD231PD(ZMM(R6), ZMM(1), MEM(RCX,R15,1)) \
    VFMADD231PD(ZMM(R7), ZMM(1), MEM(RCX,R13,2)) \
    VFMADD231PD(ZMM(R8), ZMM(1), MEM(RCX,R10,1)) \
    VMOVUPD(MEM(RCX      ), ZMM(R1)) \
    VMOVUPD(MEM(RCX,RBX,1), ZMM(R2)) \
    VMOVUPD(MEM(RCX,RBX,2), ZMM(R3)) \
    VMOVUPD(MEM(RCX,R13,1), ZMM(R4)) \
    VMOVUPD(MEM(RCX,RBX,4), ZMM(R5)) \
    VMOVUPD(MEM(RCX,R15,1), ZMM(R6)) \
    VMOVUPD(MEM(RCX,R13,2), ZMM(R7)) \
    VMOVUPD(MEM(RCX,R10,1), ZMM(R8))

#define UPDATE_C_TRANS8X8_BZ(R1,R2,R3,R4,R5,R6,R7,R8) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VMULPD(ZMM(R5), ZMM(R5), ZMM(0)) \
    VMULPD(ZMM(R6), ZMM(R6), ZMM(0)) \
    VMULPD(ZMM(R7), ZMM(R7), ZMM(0)) \
    VMULPD(ZMM(R8), ZMM(R8), ZMM(0)) \
    /* 00 01 02 03 04 05 06 07 - R1 */ \
    /* 10 11 12 13 14 15 16 17 - R2 */ \
    /* 20 21 22 23 24 25 26 27 - R3 */ \
    /* 30 31 32 33 34 35 36 37 - R4 */ \
    /* 40 41 42 43 44 45 46 47 - R5 */ \
    /* 50 51 52 53 54 55 56 57 - R6 */ \
    /* 60 61 62 63 64 65 66 67 - R7 */ \
    /* 70 71 72 73 74 75 76 77 - R8 */ \
    VUNPCKLPD(ZMM( 2), ZMM(R1), ZMM(R2)) \
    VUNPCKHPD(ZMM(R2), ZMM(R1), ZMM(R2)) \
    VUNPCKLPD(ZMM( 3), ZMM(R3), ZMM(R4)) \
    VUNPCKHPD(ZMM(R4), ZMM(R3), ZMM(R4)) \
    VUNPCKLPD(ZMM( 4), ZMM(R5), ZMM(R6)) \
    VUNPCKHPD(ZMM(R6), ZMM(R5), ZMM(R6)) \
    VUNPCKLPD(ZMM( 5), ZMM(R7), ZMM(R8)) \
    VUNPCKHPD(ZMM(R8), ZMM(R7), ZMM(R8)) \
    /* 00 10 02 12 04 14 06 16 -  2 */ \
    /* 01 11 03 13 05 15 07 17 - R2 */ \
    /* 20 30 22 32 24 34 26 36 -  3 */ \
    /* 21 31 23 33 25 35 27 37 - R4 */ \
    /* 40 50 42 52 44 54 46 56 -  4 */ \
    /* 41 51 43 53 45 55 47 57 - R6 */ \
    /* 60 70 62 72 64 74 66 76 -  5 */ \
    /* 61 71 63 73 65 75 67 77 - R8 */ \
    VSHUFF64X2(ZMM(R1), ZMM( 2), ZMM( 3), IMM(0x44)) \
    VSHUFF64X2(ZMM( 3), ZMM( 2), ZMM( 3), IMM(0xee)) \
    VSHUFF64X2(ZMM(R7), ZMM( 4), ZMM( 5), IMM(0xee)) \
    VSHUFF64X2(ZMM( 4), ZMM( 4), ZMM( 5), IMM(0x44)) \
    VSHUFF64X2(ZMM( 5), ZMM(R6), ZMM(R8), IMM(0xee)) \
    VSHUFF64X2(ZMM(R4), ZMM(R6), ZMM(R8), IMM(0x44)) \
    VSHUFF64X2(ZMM( 2), ZMM(R2), ZMM(R4), IMM(0x44)) \
    VSHUFF64X2(ZMM(R6), ZMM(R2), ZMM(R4), IMM(0xee)) \
    /* 00 10 02 12 20 30 22 32 - R1 */ \
    /* 01 11 03 13 21 31 23 33 -  2 */ \
    /* 04 14 06 16 24 34 26 36 -  3 */ \
    /* 05 15 07 17 25 35 27 37 - R6 */ \
    /* 40 50 42 52 60 70 62 72 -  4 */ \
    /* 41 51 43 53 61 71 63 73 - R4 */ \
    /* 44 54 46 56 64 74 66 76 - R7 */ \
    /* 45 55 47 57 65 75 67 77 -  5 */ \
    VSHUFF64X2(ZMM(R3), ZMM(R1), ZMM( 4), IMM(0xdd)) \
    VSHUFF64X2(ZMM(R1), ZMM(R1), ZMM( 4), IMM(0x88)) \
    VSHUFF64X2(ZMM(R2), ZMM( 2), ZMM(R4), IMM(0x88)) \
    VSHUFF64X2(ZMM(R4), ZMM( 2), ZMM(R4), IMM(0xdd)) \
    VSHUFF64X2(ZMM(R5), ZMM( 3), ZMM(R7), IMM(0x88)) \
    VSHUFF64X2(ZMM(R7), ZMM( 3), ZMM(R7), IMM(0xdd)) \
    VSHUFF64X2(ZMM(R8), ZMM(R6), ZMM( 5), IMM(0xdd)) \
    VSHUFF64X2(ZMM(R6), ZMM(R6), ZMM( 5), IMM(0x88)) \
    /* 00 10 20 30 40 50 60 70 - R1 */ \
    /* 01 11 21 31 41 51 61 71 - R2 */ \
    /* 04 14 24 34 44 54 64 74 - R5 */ \
    /* 05 15 25 35 45 55 65 75 - R6 */ \
    /* 02 12 22 32 42 52 62 72 - R3 */ \
    /* 03 13 23 33 43 53 63 73 - R4 */ \
    /* 06 16 26 36 46 56 66 76 - R7 */ \
    /* 07 17 27 37 47 57 67 77 - R8 */ \
    VMOVUPD(MEM(RCX      ), ZMM(R1)) \
    VMOVUPD(MEM(RCX,RBX,1), ZMM(R2)) \
    VMOVUPD(MEM(RCX,RBX,2), ZMM(R3)) \
    VMOVUPD(MEM(RCX,R13,1), ZMM(R4)) \
    VMOVUPD(MEM(RCX,RBX,4), ZMM(R5)) \
    VMOVUPD(MEM(RCX,R15,1), ZMM(R6)) \
    VMOVUPD(MEM(RCX,R13,2), ZMM(R7)) \
    VMOVUPD(MEM(RCX,R10,1), ZMM(R8))

#define PREFETCH_A_L1_1(n) PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*24*8))
#define PREFETCH_A_L1_2(n) PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*24*8+64))
#define PREFETCH_A_L1_3(n) PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*24*8+128))

#define PREFETCH_B_L1(n) PREFETCH(0, MEM(RBX,(B_L1_PREFETCH_DIST+n)*8*8))

//
// n: index in unrolled loop
//
// a: ZMM register to load into
// b: ZMM register to read from
//
// ...: addressing for A, except for offset
//
#define SUBITER(n,a,b,...) \
\
        VMOVAPD(ZMM(a), MEM(RBX,(n+1)*64)) \
        VFMADD231PD(ZMM( 8), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 0)*8)) \
        VFMADD231PD(ZMM( 9), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 1)*8)) \
        VFMADD231PD(ZMM(10), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 2)*8)) \
        PREFETCH_A_L1_1(n) \
        VFMADD231PD(ZMM(11), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 3)*8)) \
        VFMADD231PD(ZMM(12), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 4)*8)) \
        VFMADD231PD(ZMM(13), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 5)*8)) \
        VFMADD231PD(ZMM(14), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 6)*8)) \
        VFMADD231PD(ZMM(15), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 7)*8)) \
        VFMADD231PD(ZMM(16), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 8)*8)) \
        PREFETCH_A_L1_2(n) \
        VFMADD231PD(ZMM(17), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 9)*8)) \
        VFMADD231PD(ZMM(18), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+10)*8)) \
        VFMADD231PD(ZMM(19), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+11)*8)) \
        VFMADD231PD(ZMM(20), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+12)*8)) \
        VFMADD231PD(ZMM(21), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+13)*8)) \
        VFMADD231PD(ZMM(22), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+14)*8)) \
        PREFETCH_A_L1_3(n) \
        VFMADD231PD(ZMM(23), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+15)*8)) \
        VFMADD231PD(ZMM(24), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+16)*8)) \
        VFMADD231PD(ZMM(25), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+17)*8)) \
        VFMADD231PD(ZMM(26), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+18)*8)) \
        VFMADD231PD(ZMM(27), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+19)*8)) \
        VFMADD231PD(ZMM(28), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+20)*8)) \
        PREFETCH_B_L1(n) \
        VFMADD231PD(ZMM(29), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+21)*8)) \
        VFMADD231PD(ZMM(30), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+22)*8)) \
        VFMADD231PD(ZMM(31), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+23)*8))

//This is an array used for the scatter/gather instructions.
static int32_t offsets[32] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};

//#define MONITORS
//#define LOOPMON
void bli_dgemm_opt_24x8(
                    dim_t            k_,
                    double* restrict alpha,
                    double* restrict a,
                    double* restrict b,
                    double* restrict beta,
                    double* restrict c, inc_t rs_c_, inc_t cs_c_,
                    auxinfo_t*       data,
                    cntx_t* restrict cntx
                  )
{
    (void)data;
    (void)cntx;

    const double * a_next = bli_auxinfo_next_a( data );
    const double * b_next = bli_auxinfo_next_b( data );

    const int32_t * offsetPtr = &offsets[0];
    const int64_t k = k_;
    const int64_t rs_c = rs_c_;
    const int64_t cs_c = cs_c_;

    BEGIN_ASM

    VPXORD(ZMM(8), ZMM(8), ZMM(8)) //clear out registers
    VMOVAPD(ZMM( 9), ZMM(8))
    VMOVAPD(ZMM(10), ZMM(8))
    VMOVAPD(ZMM(11), ZMM(8))
    VMOVAPD(ZMM(12), ZMM(8))
    VMOVAPD(ZMM(13), ZMM(8))
    VMOVAPD(ZMM(14), ZMM(8))
    VMOVAPD(ZMM(15), ZMM(8))
    VMOVAPD(ZMM(16), ZMM(8))
    VMOVAPD(ZMM(17), ZMM(8))
    VMOVAPD(ZMM(18), ZMM(8))
    VMOVAPD(ZMM(19), ZMM(8))
    VMOVAPD(ZMM(20), ZMM(8))
    VMOVAPD(ZMM(21), ZMM(8))
    VMOVAPD(ZMM(22), ZMM(8))
    VMOVAPD(ZMM(23), ZMM(8))
    VMOVAPD(ZMM(24), ZMM(8))
    VMOVAPD(ZMM(25), ZMM(8))
    VMOVAPD(ZMM(26), ZMM(8))
    VMOVAPD(ZMM(27), ZMM(8))
    VMOVAPD(ZMM(28), ZMM(8))
    VMOVAPD(ZMM(29), ZMM(8))
    VMOVAPD(ZMM(30), ZMM(8))
    VMOVAPD(ZMM(31), ZMM(8))

    MOV(R12, VAR(rs_c))
    MOV(RSI, VAR(k)) //loop index
    MOV(RAX, VAR(a)) //load address of a
    MOV(RBX, VAR(b)) //load address of b
    MOV(RCX, VAR(c)) //load address of c
    VMOVAPD(ZMM(0), MEM(RBX)) //pre-load b

    MOV(RDI, VAR(offsetPtr))
    VMOVDQA32(ZMM(4), MEM(RDI))
    VMOVDQA32(ZMM(3), MEM(RDI,64))

    MOV(R8, IMM(4*24*8))     //offset for 4 iterations
    LEA(R9, MEM(R8,R8,2))    //*3
    LEA(R10, MEM(R8,R8,4))   //*5
    LEA(R11, MEM(R9,R8,4))   //*7

    //prefetch C into L2

    CMP(R12, IMM(1))
    JE(COLSTORPF2)

        KXNORW(K(1), K(0), K(0))
        KXNORW(K(2), K(0), K(0))
        VPBROADCASTD(ZMM(5), VAR(rs_c))
        VPMULLD(ZMM(2), ZMM(4), ZMM(5))
        VPMULLD(YMM(3), YMM(3), YMM(5))
        VSCATTERPFDPS(1, MEM(RCX,ZMM(2),8,64) MASK_K(1))
        VSCATTERPFDPD(1, MEM(RCX,YMM(3),8,64) MASK_K(2))

    JMP(PFDONE2)
    LABEL(COLSTORPF2)

        KXNORW(K(1), K(0), K(0))
        KXNORW(K(2), K(0), K(0))
        KXNORW(K(3), K(0), K(0))
        VPBROADCASTD(ZMM(5), VAR(cs_c))
        VPMULLD(ZMM(2), ZMM(3), ZMM(5))
        VSCATTERPFDPD(1, MEM(RCX,YMM(2),8,0*64) MASK_K(1))
        VSCATTERPFDPD(1, MEM(RCX,YMM(2),8,1*64) MASK_K(2))
        VSCATTERPFDPD(1, MEM(RCX,YMM(2),8,2*64) MASK_K(3))

    LABEL(PFDONE2)

    MOV(RDI, RSI)
    AND(RDI, IMM(31))
    SAR(RSI, IMM(5))
    JZ(REM_1)

    LOOP_ALIGN
    LABEL(MAIN_LOOP)

        SUBITER( 0,1,0,RAX      )
        SUBITER( 1,0,1,RAX      )
        SUBITER( 2,1,0,RAX      )
        SUBITER( 3,0,1,RAX      )
        SUBITER( 4,1,0,RAX,R8, 1)
        SUBITER( 5,0,1,RAX,R8, 1)
        SUBITER( 6,1,0,RAX,R8, 1)
        SUBITER( 7,0,1,RAX,R8, 1)
        SUBITER( 8,1,0,RAX,R8, 2)
        SUBITER( 9,0,1,RAX,R8, 2)
        SUBITER(10,1,0,RAX,R8, 2)
        SUBITER(11,0,1,RAX,R8, 2)
        SUBITER(12,1,0,RAX,R9, 1)
        SUBITER(13,0,1,RAX,R9, 1)
        SUBITER(14,1,0,RAX,R9, 1)
        SUBITER(15,0,1,RAX,R9, 1)
        SUBITER(16,1,0,RAX,R8, 4)
        SUBITER(17,0,1,RAX,R8, 4)
        SUBITER(18,1,0,RAX,R8, 4)
        SUBITER(19,0,1,RAX,R8, 4)
        SUBITER(20,1,0,RAX,R10,1)
        SUBITER(21,0,1,RAX,R10,1)
        SUBITER(22,1,0,RAX,R10,1)
        SUBITER(23,0,1,RAX,R10,1)
        SUBITER(24,1,0,RAX,R9, 2)
        SUBITER(25,0,1,RAX,R9, 2)
        SUBITER(26,1,0,RAX,R9, 2)
        SUBITER(27,0,1,RAX,R9, 2)
        SUBITER(28,1,0,RAX,R11,1)
        SUBITER(29,0,1,RAX,R11,1)
        SUBITER(30,1,0,RAX,R11,1)
        SUBITER(31,0,1,RAX,R11,1)

        ADD(RAX, IMM(32*24*8))
        ADD(RBX, IMM(32* 8*8))

        SUB(RSI, IMM(1))

    JNZ(MAIN_LOOP)

    //prefetch C into L1

    CMP(R12, IMM(1))
    JE(COLSTORPF)

        KXNORW(K(1), K(0), K(0))
        KXNORW(K(2), K(0), K(0))
        VSCATTERPFDPS(0, MEM(RCX,ZMM(2),8) MASK_K(1))
        VSCATTERPFDPD(0, MEM(RCX,YMM(3),8) MASK_K(2))

    JMP(PFDONE)
    LABEL(COLSTORPF)

        KXNORW(K(1), K(0), K(0))
        KXNORW(K(2), K(0), K(0))
        KXNORW(K(3), K(0), K(0))
        VPMULLD(ZMM(2), ZMM(4), ZMM(5))
        VSCATTERPFDPD(0, MEM(RCX,YMM(2),8,0*64) MASK_K(1))
        VSCATTERPFDPD(0, MEM(RCX,YMM(2),8,1*64) MASK_K(2))
        VSCATTERPFDPD(0, MEM(RCX,YMM(2),8,2*64) MASK_K(3))

    LABEL(PFDONE)

    // unrolled cleanup loop

    LABEL(REM_1)
    SAR(RDI)
    JNC(REM_2)

    SUBITER(0,1,0,RAX)
    VMOVAPD(ZMM(0), ZMM(1))
    ADD(RAX, IMM(24*8))
    ADD(RBX, IMM( 8*8))

    LABEL(REM_2)
    SAR(RDI)
    JNC(REM_4)

    SUBITER(0,1,0,RAX)
    SUBITER(1,0,1,RAX)
    ADD(RAX, IMM(2*24*8))
    ADD(RBX, IMM(2* 8*8))

    LABEL(REM_4)
    SAR(RDI)
    JNC(REM_8)

    SUBITER(0,1,0,RAX)
    SUBITER(1,0,1,RAX)
    SUBITER(2,1,0,RAX)
    SUBITER(3,0,1,RAX)
    ADD(RAX, IMM(4*24*8))
    ADD(RBX, IMM(4* 8*8))

    LABEL(REM_8)
    SAR(RDI)
    JNC(REM_16)

    SUBITER(0,1,0,RAX     )
    SUBITER(1,0,1,RAX     )
    SUBITER(2,1,0,RAX     )
    SUBITER(3,0,1,RAX     )
    SUBITER(4,1,0,RAX,R8,1)
    SUBITER(5,0,1,RAX,R8,1)
    SUBITER(6,1,0,RAX,R8,1)
    SUBITER(7,0,1,RAX,R8,1)
    ADD(RAX, IMM(8*24*8))
    ADD(RBX, IMM(8* 8*8))

    LABEL(REM_16)
    SAR(RDI)
    JNC(POSTACCUM)

    SUBITER( 0,1,0,RAX      )
    SUBITER( 1,0,1,RAX      )
    SUBITER( 2,1,0,RAX      )
    SUBITER( 3,0,1,RAX      )
    SUBITER( 4,1,0,RAX,R8, 1)
    SUBITER( 5,0,1,RAX,R8, 1)
    SUBITER( 6,1,0,RAX,R8, 1)
    SUBITER( 7,0,1,RAX,R8, 1)
    SUBITER( 8,1,0,RAX,R8, 2)
    SUBITER( 9,0,1,RAX,R8, 2)
    SUBITER(10,1,0,RAX,R8, 2)
    SUBITER(11,0,1,RAX,R8, 2)
    SUBITER(12,1,0,RAX,R9, 1)
    SUBITER(13,0,1,RAX,R9, 1)
    SUBITER(14,1,0,RAX,R9, 1)
    SUBITER(15,0,1,RAX,R9, 1)
    ADD(RAX, IMM(16*24*8))
    ADD(RBX, IMM(16* 8*8))

    LABEL(POSTACCUM)

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSD(ZMM(0), MEM(RAX))
    VBROADCASTSD(ZMM(1), MEM(RBX))

    MOV(RAX, VAR(rs_c))
    LEA(RAX, MEM(,RAX,8))
    MOV(RBX, VAR(cs_c))
    LEA(RBX, MEM(,RBX,8))
    LEA(RDI, MEM(RAX,RAX,2))

    // Check if C is row stride.
    CMP(RBX, IMM(8))
    JNE(COLSTORED)

        VMOVQ(RDX, XMM(1))
        SAL(RDX) //shift out sign bit
        JZ(ROWSTORBZ)

            UPDATE_C_FOUR_ROWS( 8, 9,10,11)
            UPDATE_C_FOUR_ROWS(12,13,14,15)
            UPDATE_C_FOUR_ROWS(16,17,18,19)
            UPDATE_C_FOUR_ROWS(20,21,22,23)
            UPDATE_C_FOUR_ROWS(24,25,26,27)
            UPDATE_C_FOUR_ROWS(28,29,30,31)

        JMP(END)
        LABEL(ROWSTORBZ)

            UPDATE_C_BZ_FOUR_ROWS( 8, 9,10,11)
            UPDATE_C_BZ_FOUR_ROWS(12,13,14,15)
            UPDATE_C_BZ_FOUR_ROWS(16,17,18,19)
            UPDATE_C_BZ_FOUR_ROWS(20,21,22,23)
            UPDATE_C_BZ_FOUR_ROWS(24,25,26,27)
            UPDATE_C_BZ_FOUR_ROWS(28,29,30,31)

    JMP(END)
    LABEL(COLSTORED)

    // Check if C is column stride. If not, jump to the slow scattered update
    CMP(RAX, IMM(8))
    JNE(SCATTEREDUPDATE)

        LEA(R13, MEM(RBX,RBX,2))
        LEA(R15, MEM(RBX,RBX,4))
        LEA(R10, MEM(R15,RBX,2))

        MOV(ESI, IMM(0x33))
        KMOV(K(1), ESI)

        VMOVQ(RDX, XMM(1))
        SAL(RDX) //shift out sign bit
        JZ(COLSTORBZ)

            UPDATE_C_TRANS8X8( 8, 9,10,11,12,13,14,15)
            ADD(RCX, IMM(64))
            UPDATE_C_TRANS8X8(16,17,18,19,20,21,22,23)
            ADD(RCX, IMM(64))
            UPDATE_C_TRANS8X8(24,25,26,27,28,29,30,31)

        JMP(END)
        LABEL(COLSTORBZ)

            UPDATE_C_TRANS8X8_BZ( 8, 9,10,11,12,13,14,15)
            ADD(RCX, IMM(64))
            UPDATE_C_TRANS8X8_BZ(16,17,18,19,20,21,22,23)
            ADD(RCX, IMM(64))
            UPDATE_C_TRANS8X8_BZ(24,25,26,27,28,29,30,31)

    JMP(END)
    LABEL(SCATTEREDUPDATE)

        VPBROADCASTD(ZMM(5), VAR(rs_c))
        VPMULLD(ZMM(2), ZMM(4), ZMM(5))

        VMOVQ(RDX, XMM(1))
        SAL(RDX) //shift out sign bit
        JZ(SCATTERBZ)

            UPDATE_C_ROW_SCATTERED( 8)
            UPDATE_C_ROW_SCATTERED( 9)
            UPDATE_C_ROW_SCATTERED(10)
            UPDATE_C_ROW_SCATTERED(11)
            UPDATE_C_ROW_SCATTERED(12)
            UPDATE_C_ROW_SCATTERED(13)
            UPDATE_C_ROW_SCATTERED(14)
            UPDATE_C_ROW_SCATTERED(15)
            UPDATE_C_ROW_SCATTERED(16)
            UPDATE_C_ROW_SCATTERED(17)
            UPDATE_C_ROW_SCATTERED(18)
            UPDATE_C_ROW_SCATTERED(19)
            UPDATE_C_ROW_SCATTERED(20)
            UPDATE_C_ROW_SCATTERED(21)
            UPDATE_C_ROW_SCATTERED(22)
            UPDATE_C_ROW_SCATTERED(23)
            UPDATE_C_ROW_SCATTERED(24)
            UPDATE_C_ROW_SCATTERED(25)
            UPDATE_C_ROW_SCATTERED(26)
            UPDATE_C_ROW_SCATTERED(27)
            UPDATE_C_ROW_SCATTERED(28)
            UPDATE_C_ROW_SCATTERED(29)
            UPDATE_C_ROW_SCATTERED(30)
            UPDATE_C_ROW_SCATTERED(31)

        JMP(END)
        LABEL(SCATTERBZ)

            UPDATE_C_BZ_ROW_SCATTERED( 8)
            UPDATE_C_BZ_ROW_SCATTERED( 9)
            UPDATE_C_BZ_ROW_SCATTERED(10)
            UPDATE_C_BZ_ROW_SCATTERED(11)
            UPDATE_C_BZ_ROW_SCATTERED(12)
            UPDATE_C_BZ_ROW_SCATTERED(13)
            UPDATE_C_BZ_ROW_SCATTERED(14)
            UPDATE_C_BZ_ROW_SCATTERED(15)
            UPDATE_C_BZ_ROW_SCATTERED(16)
            UPDATE_C_BZ_ROW_SCATTERED(17)
            UPDATE_C_BZ_ROW_SCATTERED(18)
            UPDATE_C_BZ_ROW_SCATTERED(19)
            UPDATE_C_BZ_ROW_SCATTERED(20)
            UPDATE_C_BZ_ROW_SCATTERED(21)
            UPDATE_C_BZ_ROW_SCATTERED(22)
            UPDATE_C_BZ_ROW_SCATTERED(23)
            UPDATE_C_BZ_ROW_SCATTERED(24)
            UPDATE_C_BZ_ROW_SCATTERED(25)
            UPDATE_C_BZ_ROW_SCATTERED(26)
            UPDATE_C_BZ_ROW_SCATTERED(27)
            UPDATE_C_BZ_ROW_SCATTERED(28)
            UPDATE_C_BZ_ROW_SCATTERED(29)
            UPDATE_C_BZ_ROW_SCATTERED(30)
            UPDATE_C_BZ_ROW_SCATTERED(31)

    LABEL(END)

    END_ASM
    (
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
          [a_next]    "m" (a_next),
          [b_next]    "m" (b_next),
          [offsetPtr] "m" (offsetPtr)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12",
          "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
          "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
          "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
          "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
          "zmm30", "zmm31", "memory"
    )
}
