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
#include "util/asm_x86.h"

#define A_L1_PREFETCH_DIST 4
#define B_L1_PREFETCH_DIST 4
#define TAIL_NITER 8

#define PREFETCH_A_L1(n, k) \
    PREFETCH(0, MEM(RAX, A_L1_PREFETCH_DIST*12*8 + (2*n+k)*48))
#define PREFETCH_B_L1(n, k) \
    PREFETCH(0, MEM(RBX, B_L1_PREFETCH_DIST*16*8 + (2*n+k)*64))

#define LOOP_ALIGN ALIGN16

#define UPDATE_C(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VFMADD231PD(ZMM(R1), ZMM(1), MEM(RCX,0*64)) \
    VFMADD231PD(ZMM(R2), ZMM(1), MEM(RCX,1*64)) \
    VFMADD231PD(ZMM(R3), ZMM(1), MEM(RCX,RAX,1,0*64)) \
    VFMADD231PD(ZMM(R4), ZMM(1), MEM(RCX,RAX,1,1*64)) \
    VMOVUPD(MEM(RCX,0*64), ZMM(R1)) \
    VMOVUPD(MEM(RCX,1*64), ZMM(R2)) \
    VMOVUPD(MEM(RCX,RAX,1,0*64), ZMM(R3)) \
    VMOVUPD(MEM(RCX,RAX,1,1*64), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,2))

#define UPDATE_C_BZ(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VMOVUPD(MEM(RCX,0*64), ZMM(R1)) \
    VMOVUPD(MEM(RCX,1*64), ZMM(R2)) \
    VMOVUPD(MEM(RCX,RAX,1,0*64), ZMM(R3)) \
    VMOVUPD(MEM(RCX,RAX,1,1*64), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,2))

#define UPDATE_C_ROW_SCATTERED(R1,R2,R3,R4) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    KXNORW(K(3), K(0), K(0)) \
    KXNORW(K(4), K(0), K(0)) \
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VGATHERQPD(ZMM(6) MASK_K(1), MEM(RCX,ZMM(2),1)) \
    VGATHERQPD(ZMM(6) MASK_K(2), MEM(RCX,ZMM(3),1)) \
    VFMADD231PD(ZMM(R1), ZMM(6), ZMM(1)) \
    VFMADD231PD(ZMM(R2), ZMM(6), ZMM(1)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),1) MASK_K(3), ZMM(R1)) \
    VSCATTERQPD(MEM(RCX,ZMM(3),1) MASK_K(4), ZMM(R2)) \
    LEA(RCX, MEM(RCX,RAX,1)) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    KXNORW(K(3), K(0), K(0)) \
    KXNORW(K(4), K(0), K(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VGATHERQPD(ZMM(6) MASK_K(1), MEM(RCX,ZMM(2),1)) \
    VGATHERQPD(ZMM(6) MASK_K(2), MEM(RCX,ZMM(3),1)) \
    VFMADD231PD(ZMM(R3), ZMM(6), ZMM(1)) \
    VFMADD231PD(ZMM(R4), ZMM(6), ZMM(1)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),1) MASK_K(3), ZMM(R3)) \
    VSCATTERQPD(MEM(RCX,ZMM(3),1) MASK_K(4), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,1))

#define UPDATE_C_BZ_ROW_SCATTERED(R1,R2,R3,R4) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),1) MASK_K(1), ZMM(R1)) \
    VSCATTERQPD(MEM(RCX,ZMM(3),1) MASK_K(2), ZMM(R2)) \
    LEA(RCX, MEM(RCX,RAX,1)) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),1) MASK_K(1), ZMM(R3)) \
    VSCATTERQPD(MEM(RCX,ZMM(3),1) MASK_K(2), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,1))

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

#define UPDATE_C_TRANS8X4(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    /* 00 01 02 03 04 05 06 07 - R1 */ \
    /* 10 11 12 13 14 15 16 17 - R2 */ \
    /* 20 21 22 23 24 25 26 27 - R3 */ \
    /* 30 31 32 33 34 35 36 37 - R4 */ \
    VUNPCKLPD(ZMM(2), ZMM(R1), ZMM(R2)) \
    VUNPCKHPD(ZMM(3), ZMM(R1), ZMM(R2)) \
    VUNPCKHPD(ZMM(4), ZMM(R3), ZMM(R4)) \
    VUNPCKLPD(ZMM(5), ZMM(R3), ZMM(R4)) \
    /* 00 10 02 12 04 14 06 16 -  2 */ \
    /* 01 11 03 13 05 15 07 17 -  3 */ \
    /* 20 30 22 32 24 34 26 36 -  4 */ \
    /* 21 31 23 33 25 35 27 37 -  5 */ \
    VSHUFF64X2(ZMM(R1), ZMM(2), ZMM(4), IMM(0x88)) \
    VSHUFF64X2(ZMM(R2), ZMM(3), ZMM(5), IMM(0x88)) \
    VSHUFF64X2(ZMM(R3), ZMM(2), ZMM(4), IMM(0xdd)) \
    VSHUFF64X2(ZMM(R4), ZMM(3), ZMM(5), IMM(0xdd)) \
    /* 00 10 20 30 04 14 24 34 - R1 */ \
    /* 01 11 21 31 05 15 25 35 - R2 */ \
    /* 02 12 22 32 06 16 26 36 - R3 */ \
    /* 03 13 23 33 07 17 27 37 - R4 */ \
    VMOVUPD(YMM(2), MEM(RCX,      64)) \
    VMOVUPD(YMM(3), MEM(RCX,RBX,1,64)) \
    VMOVUPD(YMM(4), MEM(RCX,RBX,2,64)) \
    VMOVUPD(YMM(5), MEM(RCX,R13,1,64)) \
    VINSERTF64X4(ZMM(2), ZMM(2), MEM(RCX,RBX,4,64), IMM(1)) \
    VINSERTF64X4(ZMM(3), ZMM(3), MEM(RCX,R15,1,64), IMM(1)) \
    VINSERTF64X4(ZMM(4), ZMM(4), MEM(RCX,R13,2,64), IMM(1)) \
    VINSERTF64X4(ZMM(5), ZMM(5), MEM(RCX,R10,1,64), IMM(1)) \
    VFMADD231PD(ZMM(R1), ZMM(1), ZMM(2)) \
    VFMADD231PD(ZMM(R2), ZMM(1), ZMM(3)) \
    VFMADD231PD(ZMM(R3), ZMM(1), ZMM(4)) \
    VFMADD231PD(ZMM(R4), ZMM(1), ZMM(5)) \
    VMOVUPD(MEM(RCX,      64), YMM(R1)) \
    VMOVUPD(MEM(RCX,RBX,1,64), YMM(R2)) \
    VMOVUPD(MEM(RCX,RBX,2,64), YMM(R3)) \
    VMOVUPD(MEM(RCX,R13,1,64), YMM(R4)) \
    VEXTRACTF64X4(MEM(RCX,RBX,4,64), ZMM(R1), IMM(1)) \
    VEXTRACTF64X4(MEM(RCX,R15,1,64), ZMM(R2), IMM(1)) \
    VEXTRACTF64X4(MEM(RCX,R13,2,64), ZMM(R3), IMM(1)) \
    VEXTRACTF64X4(MEM(RCX,R10,1,64), ZMM(R4), IMM(1))

#define UPDATE_C_TRANS8X4_BZ(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    /* 00 01 02 03 04 05 06 07 - R1 */ \
    /* 10 11 12 13 14 15 16 17 - R2 */ \
    /* 20 21 22 23 24 25 26 27 - R3 */ \
    /* 30 31 32 33 34 35 36 37 - R4 */ \
    VUNPCKLPD(ZMM(2), ZMM(R1), ZMM(R2)) \
    VUNPCKHPD(ZMM(3), ZMM(R1), ZMM(R2)) \
    VUNPCKHPD(ZMM(4), ZMM(R3), ZMM(R4)) \
    VUNPCKLPD(ZMM(5), ZMM(R3), ZMM(R4)) \
    /* 00 10 02 12 04 14 06 16 -  2 */ \
    /* 01 11 03 13 05 15 07 17 -  3 */ \
    /* 20 30 22 32 24 34 26 36 -  4 */ \
    /* 21 31 23 33 25 35 27 37 -  5 */ \
    VSHUFF64X2(ZMM(R1), ZMM(2), ZMM(4), IMM(0x44)) \
    VSHUFF64X2(ZMM(R3), ZMM(2), ZMM(4), IMM(0xee)) \
    VSHUFF64X2(ZMM(R4), ZMM(3), ZMM(5), IMM(0xee)) \
    VSHUFF64X2(ZMM(R2), ZMM(3), ZMM(5), IMM(0x44)) \
    /* 00 10 20 30 04 14 24 34 - R1 */ \
    /* 01 11 21 31 05 15 25 35 - R2 */ \
    /* 02 12 22 32 06 16 26 36 - R3 */ \
    /* 03 13 23 33 07 17 27 37 - R4 */ \
    VMOVUPD(MEM(RCX,      64), YMM(R1)) \
    VMOVUPD(MEM(RCX,RBX,1,64), YMM(R2)) \
    VMOVUPD(MEM(RCX,RBX,2,64), YMM(R3)) \
    VMOVUPD(MEM(RCX,R13,1,64), YMM(R4)) \
    VEXTRACTF64X4(MEM(RCX,RBX,4,64), ZMM(R1), IMM(1)) \
    VEXTRACTF64X4(MEM(RCX,R15,1,64), ZMM(R2), IMM(1)) \
    VEXTRACTF64X4(MEM(RCX,R13,2,64), ZMM(R3), IMM(1)) \
    VEXTRACTF64X4(MEM(RCX,R10,1,64), ZMM(R4), IMM(1))

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
    PREFETCH_A_L1(n, 0) \
    PREFETCH_B_L1(n, 0) \
    \
    VBROADCASTSD(ZMM(3), MEM(RAX,(12*n+ 0)*8)) \
    VBROADCASTSD(ZMM(4), MEM(RAX,(12*n+ 1)*8)) \
    VFMADD231PD(ZMM( 8), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM( 9), ZMM(1), ZMM(3)) \
    VFMADD231PD(ZMM(10), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(11), ZMM(1), ZMM(4)) \
    \
    VBROADCASTSD(ZMM(3), MEM(RAX,(12*n+ 2)*8)) \
    VBROADCASTSD(ZMM(4), MEM(RAX,(12*n+ 3)*8)) \
    VFMADD231PD(ZMM(12), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM(13), ZMM(1), ZMM(3)) \
    VFMADD231PD(ZMM(14), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(15), ZMM(1), ZMM(4)) \
    \
    VBROADCASTSD(ZMM(3), MEM(RAX,(12*n+ 4)*8)) \
    VBROADCASTSD(ZMM(4), MEM(RAX,(12*n+ 5)*8)) \
    VFMADD231PD(ZMM(16), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM(17), ZMM(1), ZMM(3)) \
    VFMADD231PD(ZMM(18), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(19), ZMM(1), ZMM(4)) \
    \
    PREFETCH_A_L1(n, 1) \
    PREFETCH_B_L1(n, 1) \
    \
    VBROADCASTSD(ZMM(3), MEM(RAX,(12*n+ 6)*8)) \
    VBROADCASTSD(ZMM(4), MEM(RAX,(12*n+ 7)*8)) \
    VFMADD231PD(ZMM(20), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM(21), ZMM(1), ZMM(3)) \
    VFMADD231PD(ZMM(22), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(23), ZMM(1), ZMM(4)) \
    \
    VBROADCASTSD(ZMM(3), MEM(RAX,(12*n+ 8)*8)) \
    VBROADCASTSD(ZMM(4), MEM(RAX,(12*n+ 9)*8)) \
    VFMADD231PD(ZMM(24), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM(25), ZMM(1), ZMM(3)) \
    VFMADD231PD(ZMM(26), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(27), ZMM(1), ZMM(4)) \
    \
    VBROADCASTSD(ZMM(3), MEM(RAX,(12*n+10)*8)) \
    VBROADCASTSD(ZMM(4), MEM(RAX,(12*n+11)*8)) \
    VFMADD231PD(ZMM(28), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM(29), ZMM(1), ZMM(3)) \
    VFMADD231PD(ZMM(30), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(31), ZMM(1), ZMM(4)) \
    \
    VMOVAPD(ZMM(0), MEM(RBX,(16*n+0)*8)) \
    VMOVAPD(ZMM(1), MEM(RBX,(16*n+8)*8))

//This is an array used for the scatter/gather instructions.
static int64_t offsets[16] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15};

void bli_dgemm_opt_16x12_l2(
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

    const int64_t* offsetPtr = &offsets[0];
    const int64_t k = k_;
    const int64_t rs_c = rs_c_;
    const int64_t cs_c = cs_c_;

    BEGIN_ASM

    VXORPD(YMM(8), YMM(8), YMM(8)) //clear out registers
    VMOVAPD(YMM( 7), YMM(8))
    VMOVAPD(YMM( 9), YMM(8))
    VMOVAPD(YMM(10), YMM(8))
    VMOVAPD(YMM(11), YMM(8))
    VMOVAPD(YMM(12), YMM(8))
    VMOVAPD(YMM(13), YMM(8))
    VMOVAPD(YMM(14), YMM(8))
    VMOVAPD(YMM(15), YMM(8))
    VMOVAPD(YMM(16), YMM(8))
    VMOVAPD(YMM(17), YMM(8))
    VMOVAPD(YMM(18), YMM(8))
    VMOVAPD(YMM(19), YMM(8))
    VMOVAPD(YMM(20), YMM(8))
    VMOVAPD(YMM(21), YMM(8))
    VMOVAPD(YMM(22), YMM(8))
    VMOVAPD(YMM(23), YMM(8))
    VMOVAPD(YMM(24), YMM(8))
    VMOVAPD(YMM(25), YMM(8))
    VMOVAPD(YMM(26), YMM(8))
    VMOVAPD(YMM(27), YMM(8))
    VMOVAPD(YMM(28), YMM(8))
    VMOVAPD(YMM(29), YMM(8))
    VMOVAPD(YMM(30), YMM(8))
    VMOVAPD(YMM(31), YMM(8))

    MOV(RSI, VAR(k)) //loop index
    MOV(RAX, VAR(b)) //load address of b
    MOV(RBX, VAR(a)) //load address of a
    MOV(RCX, VAR(c)) //load address of c

    TEST(RSI, RSI)
    JZ(POSTACCUM)

    MOV(R8, IMM(12*8)) //mr*sizeof(double)
    MOV(R9, IMM(16*8)) //nr*sizeof(double)

    VMOVAPD(ZMM(0), MEM(RBX, 0*8)) //pre-load b
    VMOVAPD(ZMM(1), MEM(RBX, 8*8)) //pre-load b
    LEA(RBX, MEM(RBX,R9,1)) //adjust b for pre-load

    MOV(R12, VAR(cs_c))
    LEA(R12, MEM(,R12,8))
    MOV(R10, VAR(rs_c))
    LEA(R10, MEM(,R10,8))

    //prefetch C into L2 for the next jr iteration,
    //which is 16*cs_c elements ahead

    CMP(R12, IMM(8))
    JE(ROWSTORPF2)

        LEA(R13, MEM(R12,R12,2)) //*3
        LEA(R14, MEM(R12,R12,4)) //*5
        LEA(R15, MEM(R14,R12,2)) //*7
        LEA(RDX, MEM(RCX,R13,4)) //c + 12*cs_c
        PREFETCH(1, MEM(RDX      )) PREFETCH(1, MEM(RDX,      64))
        PREFETCH(1, MEM(RDX,R12,1)) PREFETCH(1, MEM(RDX,R12,1,64))
        PREFETCH(1, MEM(RDX,R12,2)) PREFETCH(1, MEM(RDX,R12,2,64))
        PREFETCH(1, MEM(RDX,R13,1)) PREFETCH(1, MEM(RDX,R13,1,64))
        PREFETCH(1, MEM(RDX,R12,4)) PREFETCH(1, MEM(RDX,R12,4,64))
        PREFETCH(1, MEM(RDX,R14,1)) PREFETCH(1, MEM(RDX,R14,1,64))
        PREFETCH(1, MEM(RDX,R13,2)) PREFETCH(1, MEM(RDX,R13,2,64))
        PREFETCH(1, MEM(RDX,R15,1)) PREFETCH(1, MEM(RDX,R15,1,64))
        LEA(RDX, MEM(RDX,R12,8)) //c + 20*cs_c
        PREFETCH(1, MEM(RDX      )) PREFETCH(1, MEM(RDX,      64))
        PREFETCH(1, MEM(RDX,R12,1)) PREFETCH(1, MEM(RDX,R12,1,64))
        PREFETCH(1, MEM(RDX,R12,2)) PREFETCH(1, MEM(RDX,R12,2,64))
        PREFETCH(1, MEM(RDX,R13,1)) PREFETCH(1, MEM(RDX,R13,1,64))

    JMP(PFDONE2)
    LABEL(ROWSTORPF2)

        LEA(R13, MEM(R10,R10,2)) //*3
        LEA(R14, MEM(R10,R10,4)) //*5
        LEA(R15, MEM(R14,R10,2)) //*7
        MOV(RDX, RCX)
        PREFETCH(1, MEM(RDX,      96)) PREFETCH(1, MEM(RDX,      160))
        PREFETCH(1, MEM(RDX,R10,1,96)) PREFETCH(1, MEM(RDX,R10,1,160))
        PREFETCH(1, MEM(RDX,R10,2,96)) PREFETCH(1, MEM(RDX,R10,2,160))
        PREFETCH(1, MEM(RDX,R13,1,96)) PREFETCH(1, MEM(RDX,R13,1,160))
        PREFETCH(1, MEM(RDX,R10,4,96)) PREFETCH(1, MEM(RDX,R10,1,160))
        PREFETCH(1, MEM(RDX,R14,1,96)) PREFETCH(1, MEM(RDX,R14,1,160))
        PREFETCH(1, MEM(RDX,R13,2,96)) PREFETCH(1, MEM(RDX,R13,2,160))
        PREFETCH(1, MEM(RDX,R15,1,96)) PREFETCH(1, MEM(RDX,R15,1,160))
        LEA(RDX, MEM(RDX,R10,8)) //c + 8*rs_c
        PREFETCH(1, MEM(RDX,      96)) PREFETCH(1, MEM(RDX,      160))
        PREFETCH(1, MEM(RDX,R10,1,96)) PREFETCH(1, MEM(RDX,R10,1,160))
        PREFETCH(1, MEM(RDX,R10,2,96)) PREFETCH(1, MEM(RDX,R10,2,160))
        PREFETCH(1, MEM(RDX,R13,1,96)) PREFETCH(1, MEM(RDX,R13,1,160))
        PREFETCH(1, MEM(RDX,R10,4,96)) PREFETCH(1, MEM(RDX,R10,1,160))
        PREFETCH(1, MEM(RDX,R14,1,96)) PREFETCH(1, MEM(RDX,R14,1,160))
        PREFETCH(1, MEM(RDX,R13,2,96)) PREFETCH(1, MEM(RDX,R13,2,160))
        PREFETCH(1, MEM(RDX,R15,1,96)) PREFETCH(1, MEM(RDX,R15,1,160))

    LABEL(PFDONE2)

    MOV(RDI, RSI)
    AND(RSI, IMM(3))
    SAR(RDI, IMM(2))

    SUB(RDI, IMM(0+TAIL_NITER))
    JLE(K_SMALL)

    LOOP_ALIGN
    LABEL(MAIN_LOOP)

        SUBITER(0)
        SUBITER(1)
        SUBITER(2)
        SUBITER(3)

        LEA(RAX, MEM(RAX,R8,4))
        LEA(RBX, MEM(RBX,R9,4))

        DEC(RDI)

    JNZ(MAIN_LOOP)

    LABEL(K_SMALL)

    //prefetch current C into L1

    CMP(R12, IMM(8))
    JE(ROWSTORPF)

        MOV(RDX, RCX)
        PREFETCH(0, MEM(RDX      )) PREFETCH(1, MEM(RDX,      64))
        PREFETCH(0, MEM(RDX,R12,1)) PREFETCH(1, MEM(RDX,R12,1,64))
        PREFETCH(0, MEM(RDX,R12,2)) PREFETCH(1, MEM(RDX,R12,2,64))
        PREFETCH(0, MEM(RDX,R13,1)) PREFETCH(1, MEM(RDX,R13,1,64))
        PREFETCH(0, MEM(RDX,R12,4)) PREFETCH(1, MEM(RDX,R12,4,64))
        PREFETCH(0, MEM(RDX,R14,1)) PREFETCH(1, MEM(RDX,R14,1,64))
        PREFETCH(0, MEM(RDX,R13,2)) PREFETCH(1, MEM(RDX,R13,2,64))
        PREFETCH(0, MEM(RDX,R15,1)) PREFETCH(1, MEM(RDX,R15,1,64))
        LEA(RDX, MEM(RCX,R12,8)) //c + 8*cs_c
        PREFETCH(0, MEM(RDX      )) PREFETCH(1, MEM(RDX,      64))
        PREFETCH(0, MEM(RDX,R12,1)) PREFETCH(1, MEM(RDX,R12,1,64))
        PREFETCH(0, MEM(RDX,R12,2)) PREFETCH(1, MEM(RDX,R12,2,64))
        PREFETCH(0, MEM(RDX,R13,1)) PREFETCH(1, MEM(RDX,R13,1,64))

    JMP(PFDONE)
    LABEL(ROWSTORPF)

        MOV(RDX, RCX)
        PREFETCH(0, MEM(RDX      )) PREFETCH(1, MEM(RDX,      64))
        PREFETCH(0, MEM(RDX,R10,1)) PREFETCH(1, MEM(RDX,R10,1,64))
        PREFETCH(0, MEM(RDX,R10,2)) PREFETCH(1, MEM(RDX,R10,2,64))
        PREFETCH(0, MEM(RDX,R13,1)) PREFETCH(1, MEM(RDX,R13,1,64))
        PREFETCH(0, MEM(RDX,R10,4)) PREFETCH(1, MEM(RDX,R10,1,64))
        PREFETCH(0, MEM(RDX,R14,1)) PREFETCH(1, MEM(RDX,R14,1,64))
        PREFETCH(0, MEM(RDX,R13,2)) PREFETCH(1, MEM(RDX,R13,2,64))
        PREFETCH(0, MEM(RDX,R15,1)) PREFETCH(1, MEM(RDX,R15,1,64))
        LEA(RDX, MEM(RDX,R10,8)) //c + 8*rs_c
        PREFETCH(0, MEM(RDX      )) PREFETCH(1, MEM(RDX,      64))
        PREFETCH(0, MEM(RDX,R10,1)) PREFETCH(1, MEM(RDX,R10,1,64))
        PREFETCH(0, MEM(RDX,R10,2)) PREFETCH(1, MEM(RDX,R10,2,64))
        PREFETCH(0, MEM(RDX,R13,1)) PREFETCH(1, MEM(RDX,R13,1,64))
        PREFETCH(0, MEM(RDX,R10,4)) PREFETCH(1, MEM(RDX,R10,1,64))
        PREFETCH(0, MEM(RDX,R14,1)) PREFETCH(1, MEM(RDX,R14,1,64))
        PREFETCH(0, MEM(RDX,R13,2)) PREFETCH(1, MEM(RDX,R13,2,64))
        PREFETCH(0, MEM(RDX,R15,1)) PREFETCH(1, MEM(RDX,R15,1,64))

    LABEL(PFDONE)

    ADD(RDI, IMM(0+TAIL_NITER))
    JZ(TAIL_LOOP)

    LOOP_ALIGN
    LABEL(SMALL_LOOP)

        SUBITER(0)
        SUBITER(1)
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

        SUBITER(0)

        ADD(RAX, R8)
        ADD(RBX, R9)

        DEC(RSI)

    JNZ(TAIL_LOOP)

    LABEL(POSTACCUM)

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSD(ZMM(0), MEM(RAX))
    VBROADCASTSD(ZMM(1), MEM(RBX))

    MOV(RAX, VAR(cs_c))
    LEA(RAX, MEM(,RAX,8))
    MOV(RBX, VAR(rs_c))
    LEA(RBX, MEM(,RBX,8))

    // Check if C is column stride.
    CMP(RBX, IMM(8))
    JNE(ROWSTORED)

        VCOMISD(XMM(1), XMM(7))
        JE(COLSTORBZ)

            UPDATE_C( 8, 9,10,11)
            UPDATE_C(12,13,14,15)
            UPDATE_C(16,17,18,19)
            UPDATE_C(20,21,22,23)
            UPDATE_C(24,25,26,27)
            UPDATE_C(28,29,30,31)

        JMP(END)
        LABEL(COLSTORBZ)

            UPDATE_C_BZ( 8, 9,10,11)
            UPDATE_C_BZ(12,13,14,15)
            UPDATE_C_BZ(16,17,18,19)
            UPDATE_C_BZ(20,21,22,23)
            UPDATE_C_BZ(24,25,26,27)
            UPDATE_C_BZ(28,29,30,31)

    JMP(END)
    LABEL(ROWSTORED)

    // Check if C is row stride. If not, jump to the slow scattered update
    CMP(RAX, IMM(8))
    JNE(SCATTEREDUPDATE)

        //
        // Transpose and write out in four quadrants:
        //
        // +-------------------------------+-------------------------------+
        // |c00-c01-c02-c03-c04-c05-c06-c07|c08-c09-c0A-c0B-c0C-c0D-c0E-c0F|
        // |                               |                               |
        // |c10-c11-c12-c13-c14-c15-c16-c17|c18-c19-c1A-c1B-c1C-c1D-c1E-c1F|
        // |                               |                               |
        // |c20-c21-c22-c23-c24-c25-c26-c27|c28-c29-c2A-c2B-c2C-c2D-c2E-c2F|
        // |                               |                               |
        // |c30-c31-c32-c33-c34-c35-c36-c37|c38-c39-c3A-c3B-c3C-c3D-c3E-c3F|     +------+------+
        // |                               |                               |     |      |      |
        // |c40-c41-c42-c43-c44-c45-c46-c47|c48-c49-c4A-c4B-c4C-c4D-c4E-c4F|     |  Q1  |  Q3  |
        // |                               |                               |     |      |      |
        // |c50-c51-c52-c53-c54-c55-c56-c57|c58-c59-c5A-c5B-c5C-c5D-c5E-c5F|  =  +------+------+
        // |                               |                               |     |      |      |
        // |c60-c61-c62-c63-c64-c65-c66-c67|c68-c69-c6A-c6B-c6C-c6D-c6E-c6F|     |  Q2  |  Q4  |
        // |                               |                               |     |      |      |
        // |c70-c71-c72-c73-c74-c75-c76-c77|c78-c79-c7A-c7B-c7C-c7D-c7E-c7F|     +------+------+
        // +-------------------------------+-------------------------------+
        // |c80-c81-c82-c83-c84-c85-c86-c87|c88-c89-c8A-c8B-c8C-c8D-c8E-c8F|
        // |                               |                               |
        // |c90-c91-c92-c93-c94-c95-c96-c97|c98-c99-c9A-c9B-c9C-c9D-c9E-c9F|
        // |                               |                               |
        // |cA0-cA1-cA2-cA3-cA4-cA5-cA6-cA7|cA8-cA9-cAA-cAB-cAC-cAD-cAE-cAF|
        // |                               |                               |
        // |cB0-cB1-cB2-cB3-cB4-cB5-cB6-cB7|cB8-cB9-cBA cBB-cBC-cBD-cBE-cBF|
        // +-------------------------------+-------------------------------+
        //
        //                                ||
        //                                \/
        //
        // +-------------------------------+-------------------------------+
        // |c00 c01 c02 c03 c04 c05 c06 c07|c08 c09 c0A c0B c0C c0D c0E c0F|
        // | |   |   |   |   |   |   |   | | |   |   |   |   |   |   |   | |
        // |c10 c11 c12 c13 c14 c15 c16 c17|c18 c19 c1A c1B c1C c1D c1E c1F|
        // | |   |   |   |   |   |   |   | | |   |   |   |   |   |   |   | |
        // |c20 c21 c22 c23 c24 c25 c26 c27|c28 c29 c2A c2B c2C c2D c2E c2F|
        // | |   |   |   |   |   |   |   | | |   |   |   |   |   |   |   | |
        // |c30 c31 c32 c33 c34 c35 c36 c37|c38 c39 c3A c3B c3C c3D c3E c3F|
        // | |   |   |   |   |   |   |   | | |   |   |   |   |   |   |   | |
        // |c40 c41 c42 c43 c44 c45 c46 c47|c48 c49 c4A c4B c4C c4D c4E c4F|
        // | |   |   |   |   |   |   |   | | |   |   |   |   |   |   |   | |
        // |c50 c51 c52 c53 c54 c55 c56 c57|c58 c59 c5A c5B c5C c5D c5E c5F|
        // | |   |   |   |   |   |   |   | | |   |   |   |   |   |   |   | |
        // |c60 c61 c62 c63 c64 c65 c66 c67|c68 c69 c6A c6B c6C c6D c6E c6F|
        // | |   |   |   |   |   |   |   | | |   |   |   |   |   |   |   | |
        // |c70 c71 c72 c73 c74 c75 c76 c77|c78 c79 c7A c7B c7C c7D c7E c7F|
        // +-------------------------------+-------------------------------+
        // |c80 c81 c82 c83 c84 c85 c86 c87|c88 c89 c8A c8B c8C c8D c8E c8F|
        // | |   |   |   |   |   |   |   | | |   |   |   |   |   |   |   | |
        // |c90 c91 c92 c93 c94 c95 c96 c97|c98 c99 c9A c9B c9C c9D c9E c9F|
        // | |   |   |   |   |   |   |   | | |   |   |   |   |   |   |   | |
        // |cA0 cA1 cA2 cA3 cA4 cA5 cA6 cA7|cA8 cA9 cAA cAB cAC cAD cAE cAF|
        // | |   |   |   |   |   |   |   | | |   |   |   |   |   |   |   | |
        // |cB0 cB1 cB2 cB3 cB4 cB5 cB6 cB7|cB8 cB9 cBA cBB cBC cBD cBE cBF|
        // +-------------------------------+-------------------------------+
        //

        LEA(R13, MEM(RBX,RBX,2))
        LEA(R15, MEM(RBX,RBX,4))
        LEA(R10, MEM(R15,RBX,2))

        VCOMISD(XMM(1), XMM(7))
        JE(ROWSTORBZ)

            UPDATE_C_TRANS8X8( 8,10,12,14,16,18,20,22)
            UPDATE_C_TRANS8X4(24,26,28,30)

            LEA(RCX, MEM(RCX,RBX,8))

            UPDATE_C_TRANS8X8( 9,11,13,15,17,19,21,23)
            UPDATE_C_TRANS8X4(25,27,29,31)

        JMP(END)
        LABEL(ROWSTORBZ)

            UPDATE_C_TRANS8X8_BZ( 8,10,12,14,16,18,20,22)
            UPDATE_C_TRANS8X4_BZ(24,26,28,30)

            LEA(RCX, MEM(RCX,RBX,8))

            UPDATE_C_TRANS8X8_BZ( 9,11,13,15,17,19,21,23)
            UPDATE_C_TRANS8X4_BZ(25,27,29,31)

    JMP(END)
    LABEL(SCATTEREDUPDATE)

        MOV(RDI, VAR(offsetPtr))
        VMOVDQA64(ZMM(2), MEM(RDI,0*64))
        VMOVDQA64(ZMM(3), MEM(RDI,1*64))
        VPBROADCASTQ(ZMM(6), RBX)
        VPMULLQ(ZMM(2), ZMM(6), ZMM(2))
        VPMULLQ(ZMM(3), ZMM(6), ZMM(3))

        VCOMISD(XMM(1), XMM(7))
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
