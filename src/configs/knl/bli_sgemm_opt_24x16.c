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

#define SCATTER_PREFETCH_C 1

#define PREFETCH_A_L2 0
#define PREFETCH_B_L2 0
#define L2_PREFETCH_DIST 64

#define A_L1_PREFETCH_DIST 36
#define B_L1_PREFETCH_DIST 18

#define LOOP_ALIGN ALIGN16

#define UPDATE_C_FOUR_ROWS(R1,R2,R3,R4) \
\
    VMULPS(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPS(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPS(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPS(ZMM(R4), ZMM(R4), ZMM(0)) \
    VFMADD231PS(ZMM(R1), ZMM(1), MEM(RCX      )) \
    VFMADD231PS(ZMM(R2), ZMM(1), MEM(RCX,RAX,1)) \
    VFMADD231PS(ZMM(R3), ZMM(1), MEM(RCX,RAX,2)) \
    VFMADD231PS(ZMM(R4), ZMM(1), MEM(RCX,RDI,1)) \
    VMOVUPS(MEM(RCX      ), ZMM(R1)) \
    VMOVUPS(MEM(RCX,RAX,1), ZMM(R2)) \
    VMOVUPS(MEM(RCX,RAX,2), ZMM(R3)) \
    VMOVUPS(MEM(RCX,RDI,1), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,4))

#define UPDATE_C_BZ_FOUR_ROWS(R1,R2,R3,R4) \
\
    VMULPS(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPS(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPS(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPS(ZMM(R4), ZMM(R4), ZMM(0)) \
    VMOVUPS(MEM(RCX      ), ZMM(R1)) \
    VMOVUPS(MEM(RCX,RAX,1), ZMM(R2)) \
    VMOVUPS(MEM(RCX,RAX,2), ZMM(R3)) \
    VMOVUPS(MEM(RCX,RDI,1), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,4))

#define UPDATE_C_ROW_SCATTERED(NUM) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPS(ZMM(NUM), ZMM(NUM), ZMM(0)) \
    VGATHERDPS(ZMM(3) MASK_K(1), MEM(RCX,ZMM(2),4)) \
    VFMADD231PS(ZMM(NUM), ZMM(3), ZMM(1)) \
    VSCATTERDPS(MEM(RCX,ZMM(2),4) MASK_K(2), ZMM(NUM)) \
    ADD(RCX, RAX)

#define UPDATE_C_BZ_ROW_SCATTERED(NUM) \
\
    KXNORW(K(1), K(0), K(0)) \
    VMULPS(ZMM(NUM), ZMM(NUM), ZMM(0)) \
    VSCATTERDPS(MEM(RCX,ZMM(2),4) MASK_K(1), ZMM(NUM)) \
    ADD(RCX, RAX)

#define PREFETCH_A_L1_1(n) PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*24*4))
#define PREFETCH_A_L1_2(n) PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*24*4+64))

#if PREFETCH_A_L2
#undef PREFETCH_A_L2

#define PREFETCH_A_L2(n) \
\
    PREFETCH(1, MEM(RAX,(L2_PREFETCH_DIST+n)*24*4)) \
    PREFETCH(1, MEM(RAX,(L2_PREFETCH_DIST+n)*24*4+64))

#else
#undef PREFETCH_A_L2
#define PREFETCH_A_L2(...)
#endif

#define PREFETCH_B_L1(n) PREFETCH(0, MEM(RBX,(B_L1_PREFETCH_DIST+n)*16*4))

#if PREFETCH_B_L2
#undef PREFETCH_B_L2

#define PREFETCH_B_L2(n) PREFETCH(1, MEM(RBX,(L2_PREFETCH_DIST+n)*16*4))

#else
#undef PREFETCH_B_L2
#define PREFETCH_B_L2(...)
#endif

#define PREFETCH_C_L1_1
#define PREFETCH_C_L1_2
#define PREFETCH_C_L1_3

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
        PREFETCH_A_L2(n) \
\
        VMOVAPS(ZMM(a), MEM(RBX,(n+1)*64)) \
        VFMADD231PS(ZMM( 8), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 0)*4)) \
        VFMADD231PS(ZMM( 9), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 1)*4)) \
        VFMADD231PS(ZMM(10), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 2)*4)) \
        PREFETCH_A_L1_1(n) \
        VFMADD231PS(ZMM(11), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 3)*4)) \
        VFMADD231PS(ZMM(12), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 4)*4)) \
        VFMADD231PS(ZMM(13), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 5)*4)) \
        PREFETCH_C_L1_1 \
        VFMADD231PS(ZMM(14), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 6)*4)) \
        VFMADD231PS(ZMM(15), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 7)*4)) \
        VFMADD231PS(ZMM(16), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 8)*4)) \
        PREFETCH_A_L1_2(n) \
        VFMADD231PS(ZMM(17), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 9)*4)) \
        VFMADD231PS(ZMM(18), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+10)*4)) \
        VFMADD231PS(ZMM(19), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+11)*4)) \
        PREFETCH_C_L1_2 \
        VFMADD231PS(ZMM(20), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+12)*4)) \
        VFMADD231PS(ZMM(21), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+13)*4)) \
        VFMADD231PS(ZMM(22), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+14)*4)) \
        PREFETCH_C_L1_3 \
        VFMADD231PS(ZMM(23), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+15)*4)) \
        VFMADD231PS(ZMM(24), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+16)*4)) \
        VFMADD231PS(ZMM(25), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+17)*4)) \
        PREFETCH_B_L1(n) \
        VFMADD231PS(ZMM(26), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+18)*4)) \
        VFMADD231PS(ZMM(27), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+19)*4)) \
        VFMADD231PS(ZMM(28), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+20)*4)) \
        PREFETCH_B_L2(n) \
        VFMADD231PS(ZMM(29), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+21)*4)) \
        VFMADD231PS(ZMM(30), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+22)*4)) \
        VFMADD231PS(ZMM(31), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+23)*4))

//This is an array used for the scatter/gather instructions.
static int32_t offsets[32] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};

//#define MONITORS
//#define LOOPMON
void bli_sgemm_opt_24x16(
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

#ifdef MONITORS
    int toph, topl, both, botl, midl, midh, mid2l, mid2h;
#endif
#ifdef LOOPMON
    int tlooph, tloopl, blooph, bloopl;
#endif

    __asm__ volatile
    (
#ifdef MONITORS
    RDTSC
    MOV(VAR(topl), EAX)
    MOV(VAR(toph), EDX)
#endif

    VPXORD(ZMM(8), ZMM(8), ZMM(8)) //clear out registers
    VMOVAPS(ZMM( 9), ZMM(8))   MOV(R12, VAR(rs_c))
    VMOVAPS(ZMM(10), ZMM(8))   MOV(RSI, VAR(k)) //loop index
    VMOVAPS(ZMM(11), ZMM(8))   MOV(RAX, VAR(a)) //load address of a
    VMOVAPS(ZMM(12), ZMM(8))   MOV(RBX, VAR(b)) //load address of b
    VMOVAPS(ZMM(13), ZMM(8))   MOV(RCX, VAR(c)) //load address of c
    VMOVAPS(ZMM(14), ZMM(8))   VMOVAPD(ZMM(0), MEM(RBX)) //pre-load b
    VMOVAPS(ZMM(15), ZMM(8))   MOV(RDI, VAR(offsetPtr))
    VMOVAPS(ZMM(16), ZMM(8))   VMOVAPS(ZMM(4), MEM(RDI))
#if SCATTER_PREFETCH_C
    VMOVAPS(ZMM(17), ZMM(8))
    VMOVAPS(ZMM(18), ZMM(8))
    VMOVAPS(ZMM(19), ZMM(8))   VBROADCASTSS(ZMM(5), VAR(rs_c))
    VMOVAPS(ZMM(20), ZMM(8))
    VMOVAPS(ZMM(21), ZMM(8))   VPMULLD(ZMM(2), ZMM(4), ZMM(5))
    VMOVAPS(ZMM(22), ZMM(8))   VMOVAPS(YMM(3), MEM(RDI,64))
    VMOVAPS(ZMM(23), ZMM(8))   VPMULLD(YMM(3), YMM(3), YMM(5))
#else
    VMOVAPS(ZMM(17), ZMM(8))
    VMOVAPS(ZMM(18), ZMM(8))   LEA(R13, MEM(R12,R12,2))
    VMOVAPS(ZMM(19), ZMM(8))   LEA(R14, MEM(R12,R12,4))
    VMOVAPS(ZMM(20), ZMM(8))   LEA(R15, MEM(R13,R12,4))
    VMOVAPS(ZMM(21), ZMM(8))
    VMOVAPS(ZMM(22), ZMM(8))
    VMOVAPS(ZMM(23), ZMM(8))
#endif
    VMOVAPS(ZMM(24), ZMM(8))   VPSLLD(ZMM(4), ZMM(4), IMM(2))
    VMOVAPS(ZMM(25), ZMM(8))   MOV(R8, IMM(4*24*4))     //offset for 4 iterations
    VMOVAPS(ZMM(26), ZMM(8))   LEA(R9, MEM(R8,R8,2))    //*3
    VMOVAPS(ZMM(27), ZMM(8))   LEA(R10, MEM(R8,R8,4))   //*5
    VMOVAPS(ZMM(28), ZMM(8))   LEA(R11, MEM(R9,R8,4))   //*7
    VMOVAPS(ZMM(29), ZMM(8))
    VMOVAPS(ZMM(30), ZMM(8))
    VMOVAPS(ZMM(31), ZMM(8))

#ifdef MONITORS
    RDTSC
    MOV(VAR(midl), EAX)
    MOV(VAR(midh), EDX)
#endif

    SUB(RSI, IMM(32))
    JLE(TAIL)

    //prefetch C into L2
#if SCATTER_PREFETCH_C
    ADD(RSI, IMM(24))
    KXNORW(K(1), K(0), K(0))
    KXNORW(K(2), K(0), K(0))
    VSCATTERPFDPS(1, MEM(RCX,ZMM(2),8) MASK_K(1))
    VSCATTERPFDPD(1, MEM(RCX,YMM(3),8) MASK_K(2))
#else
    PREFETCHW1(MEM(RCX      ))
    SUBITER( 0,1,0,RAX      )
    PREFETCHW1(MEM(RCX,R12,1))
    SUBITER( 1,0,1,RAX      )
    PREFETCHW1(MEM(RCX,R12,2))
    SUBITER( 2,1,0,RAX      )
    PREFETCHW1(MEM(RCX,R13,1))
    SUBITER( 3,0,1,RAX      )
    PREFETCHW1(MEM(RCX,R12,4))
    SUBITER( 4,1,0,RAX,R8, 1)
    PREFETCHW1(MEM(RCX,R14,1))
    SUBITER( 5,0,1,RAX,R8, 1)
    PREFETCHW1(MEM(RCX,R13,2))
    SUBITER( 6,1,0,RAX,R8, 1)
    PREFETCHW1(MEM(RCX,R15,1))
    SUBITER( 7,0,1,RAX,R8, 1)

    LEA(RDX, MEM(RCX,R12,8))

    PREFETCHW1(MEM(RDX      ))
    SUBITER( 8,1,0,RAX,R8, 2)
    PREFETCHW1(MEM(RDX,R12,1))
    SUBITER( 9,0,1,RAX,R8, 2)
    PREFETCHW1(MEM(RDX,R12,2))
    SUBITER(10,1,0,RAX,R8, 2)
    PREFETCHW1(MEM(RDX,R13,1))
    SUBITER(11,0,1,RAX,R8, 2)
    PREFETCHW1(MEM(RDX,R12,4))
    SUBITER(12,1,0,RAX,R9, 1)
    PREFETCHW1(MEM(RDX,R14,1))
    SUBITER(13,0,1,RAX,R9, 1)
    PREFETCHW1(MEM(RDX,R13,2))
    SUBITER(14,1,0,RAX,R9, 1)
    PREFETCHW1(MEM(RDX,R15,1))
    SUBITER(15,0,1,RAX,R9, 1)

    LEA(RDI, MEM(RDX,R12,8))

    PREFETCHW1(MEM(RDI      ))
    SUBITER(16,1,0,RAX,R8, 4)
    PREFETCHW1(MEM(RDI,R12,1))
    SUBITER(17,0,1,RAX,R8, 4)
    PREFETCHW1(MEM(RDI,R12,2))
    SUBITER(18,1,0,RAX,R8, 4)
    PREFETCHW1(MEM(RDI,R13,1))
    SUBITER(19,0,1,RAX,R8, 4)
    PREFETCHW1(MEM(RDI,R12,4))
    SUBITER(20,1,0,RAX,R10,1)
    PREFETCHW1(MEM(RDI,R14,1))
    SUBITER(21,0,1,RAX,R10,1)
    PREFETCHW1(MEM(RDI,R13,2))
    SUBITER(22,1,0,RAX,R10,1)
    PREFETCHW1(MEM(RDI,R15,1))
    SUBITER(23,0,1,RAX,R10,1)

    ADD(RAX, IMM(24*24*4))
    ADD(RBX, IMM(24*16*4))
#endif

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

        ADD(RAX, IMM(32*24*4))
        ADD(RBX, IMM(32*16*4))

        SUB(RSI, IMM(1))

    JNZ(MAIN_LOOP)

    LABEL(REM_1)
    SAR(RDI)
    JNC(REM_2)

    SUBITER(0,1,0,RAX)
    VMOVAPD(ZMM(0), ZMM(1))
    ADD(RAX, IMM(24*4))
    ADD(RBX, IMM(16*4))

    LABEL(REM_2)
    SAR(RDI)
    JNC(REM_4)

    SUBITER(0,1,0,RAX)
    SUBITER(1,0,1,RAX)
    ADD(RAX, IMM(2*24*4))
    ADD(RBX, IMM(2*16*4))

    LABEL(REM_4)
    SAR(RDI)
    JNC(REM_8)

    SUBITER(0,1,0,RAX)
    SUBITER(1,0,1,RAX)
    SUBITER(2,1,0,RAX)
    SUBITER(3,0,1,RAX)
    ADD(RAX, IMM(4*24*4))
    ADD(RBX, IMM(4*16*4))

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
    ADD(RAX, IMM(8*24*4))
    ADD(RBX, IMM(8*16*4))

    LABEL(REM_16)
    SAR(RDI)
    JNC(AFTER_LOOP)

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
    ADD(RAX, IMM(16*24*4))
    ADD(RBX, IMM(16*16*4))

    LABEL(AFTER_LOOP)

    //prefetch C into L1
#if SCATTER_PREFETCH_C
    KXNORW(K(1), K(0), K(0))
    KXNORW(K(2), K(0), K(0))
    VSCATTERPFDPS(0, MEM(RCX,ZMM(2),8) MASK_K(1))
    VSCATTERPFDPD(0, MEM(RCX,YMM(3),8) MASK_K(2))

    SUBITER(0,1,0,RAX     )
    SUBITER(1,0,1,RAX     )
    SUBITER(2,1,0,RAX     )
    SUBITER(3,0,1,RAX     )
    SUBITER(4,1,0,RAX,R8,1)
    SUBITER(5,0,1,RAX,R8,1)
    SUBITER(6,1,0,RAX,R8,1)
    SUBITER(7,0,1,RAX,R8,1)
#else

    LEA(RDX, MEM(RCX,R12,8))
    LEA(RDI, MEM(RDX,R12,8))

#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RCX      ))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RCX,R12,1))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RCX,R12,2))
    SUBITER(0,1,0,RAX     )
#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RCX,R13,1))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RCX,R12,4))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RCX,R14,1))
    SUBITER(1,0,1,RAX     )
#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RCX,R13,2))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RCX,R15,1))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RDX      ))
    SUBITER(2,1,0,RAX     )
#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RDX,R12,1))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RDX,R12,2))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RDX,R13,1))
    SUBITER(3,0,1,RAX     )
#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RDX,R12,4))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RDX,R14,1))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RDX,R13,2))
    SUBITER(4,1,0,RAX,R8,1)
#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RDX,R15,1))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RDI      ))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RDI,R12,1))
    SUBITER(5,0,1,RAX,R8,1)
#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RDI,R12,2))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RDI,R13,1))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RDI,R12,4))
    SUBITER(6,1,0,RAX,R8,1)
#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RDI,R14,1))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RDI,R13,2))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RDI,R15,1))
    SUBITER(7,0,1,RAX,R8,1)
#endif

    JMP(POSTACCUM)

    LABEL(TAIL)

    MOV(RDX, RCX)
    ADD(RSI, IMM(32))
    JZ(POSTACCUM)

    LABEL(TAIL_LOOP)

        PREFETCHW0(MEM(RDX))
        ADD(RDX, R12)

        SUBITER(0,1,0,RAX)
        VMOVAPD(ZMM(0), ZMM(1))
        ADD(RAX, IMM(24*4))
        ADD(RBX, IMM(16*4))

        SUB(RSI, IMM(1))

    JNZ(TAIL_LOOP)

    LABEL(POSTACCUM)

#ifdef MONITORS
    RDTSC
    MOV(VAR(mid2l), EAX)
    MOV(VAR(mid2h), EDX)
#endif

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSS(ZMM(0), MEM(RAX))
    VBROADCASTSS(ZMM(1), MEM(RBX))

    MOV(RAX, VAR(rs_c))
    LEA(RAX, MEM(,RAX,4))
    MOV(RBX, VAR(cs_c))
    LEA(RBX, MEM(,RBX,4))
    LEA(RDI, MEM(RAX,RAX,2))

    // Check if C is row stride.
    CMP(RBX, IMM(4))
    JNE(COLSTORED)

        VMOVD(EDX, XMM(1))
        SAL(EDX) //shift out sign bit
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
    CMP(RAX, IMM(4))
    JNE(SCATTEREDUPDATE)

        LEA(R13, MEM(RBX,RBX,2))
        LEA(R15, MEM(RBX,RBX,4))
        LEA(R10, MEM(R15,RBX,2))
        LEA(RDX, MEM(RCX,RBX,8))

        MOV(ESI, IMM(0x0f0f))
        MOV(EDI, IMM(0x00ff))
        KMOV(K(1), ESI)
        KMOV(K(2), EDI)

        VMOVD(EDI, XMM(1))
        SAL(EDI) //shift out sign bit
        JZ(COLSTORBZ)

            // Transpose and write out the last 8 rows so we can use them as temporaries

            VMULPD(ZMM(24), ZMM(24), ZMM(0))
            VMULPD(ZMM(25), ZMM(25), ZMM(0))
            VMULPD(ZMM(26), ZMM(26), ZMM(0))
            VMULPD(ZMM(27), ZMM(27), ZMM(0))
            VMULPD(ZMM(28), ZMM(28), ZMM(0))
            VMULPD(ZMM(29), ZMM(29), ZMM(0))
            VMULPD(ZMM(30), ZMM(30), ZMM(0))
            VMULPD(ZMM(31), ZMM(31), ZMM(0))
            /* 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F - 24 */
            /* 10 11 12 13 14 15 16 17 18 19 1A 1B 1C 1D 1E 1F - 25 */
            /* 20 21 22 23 24 25 26 27 28 29 2A 2B 2C 2D 2E 2F - 26 */
            /* 30 31 32 33 34 35 36 37 38 39 3A 3B 3C 3D 3E 3F - 27 */
            /* 40 41 42 43 44 45 46 47 48 49 4A 4B 4C 4D 4E 4F - 28 */
            /* 50 51 52 53 54 55 56 57 58 59 5A 5B 5C 5D 5E 5F - 29 */
            /* 60 61 62 63 64 65 66 67 68 69 6A 6B 6C 6D 6E 6F - 30 */
            /* 70 71 72 73 74 75 76 77 78 79 7A 7B 7C 7D 7E 7F - 31 */
            VUNPCKLPS(ZMM( 2), ZMM(24), ZMM(25))
            VUNPCKHPS(ZMM( 3), ZMM(26), ZMM(27))
            VUNPCKLPS(ZMM( 4), ZMM(28), ZMM(29))
            VUNPCKHPS(ZMM( 5), ZMM(30), ZMM(31))
            VUNPCKHPS(ZMM(25), ZMM(24), ZMM(25))
            VUNPCKLPS(ZMM(26), ZMM(26), ZMM(27))
            VUNPCKHPS(ZMM(29), ZMM(28), ZMM(29))
            VUNPCKLPS(ZMM(30), ZMM(30), ZMM(31))
            /* 00 10 02 12 04 14 06 16 08 18 0A 1A 0C 1C 0E 1E -  2 */
            /* 01 11 03 13 05 15 07 17 09 19 0B 1B 0D 1D 0F 1F - 25 */
            /* 20 30 22 32 24 34 26 36 28 38 2A 3A 2C 3C 2E 3E - 26 */
            /* 21 31 23 33 25 35 27 37 29 39 2B 3B 2D 3D 2F 3F -  3 */
            /* 40 50 42 52 44 54 46 56 48 58 4A 5A 4C 5C 4E 5E -  4 */
            /* 41 51 43 53 45 55 47 57 49 59 4B 5B 4D 5D 4F 5F - 29 */
            /* 60 70 62 72 64 74 66 76 68 78 6A 7A 6C 7C 6E 7E - 30 */
            /* 61 71 63 73 65 75 67 77 69 79 6B 7B 6D 7D 6F 7F -  5 */
            VSHUFPS(ZMM(24), ZMM( 2), ZMM(26), IMM(0x44))
            VSHUFPS(ZMM(27), ZMM(25), ZMM( 3), IMM(0xee))
            VSHUFPS(ZMM(28), ZMM( 4), ZMM(30), IMM(0x44))
            VSHUFPS(ZMM(31), ZMM(29), ZMM( 5), IMM(0xee))
            VSHUFPS(ZMM(26), ZMM( 2), ZMM(26), IMM(0xee))
            VSHUFPS(ZMM(25), ZMM(25), ZMM( 3), IMM(0x44))
            VSHUFPS(ZMM(30), ZMM( 4), ZMM(30), IMM(0xee))
            VSHUFPS(ZMM(29), ZMM(29), ZMM( 5), IMM(0x44))
            /* 00 10 20 30 04 14 24 34 08 18 28 38 0C 1C 2C 3C - 24 */
            /* 01 11 03 31 05 15 25 35 09 19 29 39 0D 1D 2D 3D - 25 */
            /* 02 12 22 32 06 16 26 36 0A 1A 2A 3A 0E 1E 2E 3E - 26 */
            /* 03 13 23 33 07 17 27 37 0B 1B 2B 3B 0F 1F 2F 3F - 27 */
            /* 40 50 60 70 44 54 64 74 48 58 68 78 4C 5C 6C 7C - 28 */
            /* 41 51 61 71 45 55 65 75 49 59 69 79 4D 5D 6D 7D - 29 */
            /* 42 52 62 72 46 56 66 76 4A 5A 6A 7A 4E 5E 6E 7E - 30 */
            /* 43 53 63 73 47 57 67 77 4B 5B 6B 7B 4F 5F 6F 7F - 31 */
            VBLENDMPS(ZMM(2) MASK_K(1), ZMM(24), ZMM(28))
            VPERMPD(ZMM(2), ZMM(2), IMM(0x4e))
            VBLENDMPS(ZMM(24) MASK_K(1), ZMM(2), ZMM(24))
            VBLENDMPS(ZMM(28) MASK_K(1), ZMM(28), ZMM(2))
            VBLENDMPS(ZMM(3) MASK_K(1), ZMM(25), ZMM(29))
            VPERMPD(ZMM(3), ZMM(3), IMM(0x4e))
            VBLENDMPS(ZMM(25) MASK_K(1), ZMM(3), ZMM(25))
            VBLENDMPS(ZMM(29) MASK_K(1), ZMM(29), ZMM(3))
            VBLENDMPS(ZMM(4) MASK_K(1), ZMM(26), ZMM(30))
            VPERMPD(ZMM(4), ZMM(4), IMM(0x4e))
            VBLENDMPS(ZMM(26) MASK_K(1), ZMM(4), ZMM(26))
            VBLENDMPS(ZMM(30) MASK_K(1), ZMM(30), ZMM(4))
            VBLENDMPS(ZMM(5) MASK_K(1), ZMM(27), ZMM(31))
            VPERMPD(ZMM(5), ZMM(5), IMM(0x4e))
            VBLENDMPS(ZMM(27) MASK_K(1), ZMM(5), ZMM(27))
            VBLENDMPS(ZMM(31) MASK_K(1), ZMM(31), ZMM(5))
            /* 00 10 20 30 40 50 60 70 04 14 24 34 44 54 64 74 - 24 */
            /* 01 11 03 31 41 51 61 71 05 15 25 35 45 55 65 75 - 25 */
            /* 02 12 22 32 42 52 62 72 06 16 26 36 46 56 66 76 - 26 */
            /* 03 13 23 33 43 53 63 73 07 17 27 37 47 57 67 77 - 27 */
            /* 08 18 28 38 48 58 68 78 0C 1C 2C 3C 4C 5C 6C 7C - 28 */
            /* 09 19 29 39 49 59 69 79 0D 1D 2D 3D 4D 5D 6D 7D - 29 */
            /* 0A 1A 2A 3A 4A 5A 6A 7A 0E 1E 2E 3E 4E 5E 6E 7E - 30 */
            /* 0B 1B 2B 3B 4B 5B 6B 7B 0F 1F 2F 3F 4F 5F 6F 7F - 31 */
            VMOVUPS(YMM(2), MEM(RCX,      64))
            VMOVUPS(YMM(3), MEM(RCX,RBX,1,64))
            VMOVUPS(YMM(4), MEM(RCX,RBX,2,64))
            VMOVUPS(YMM(5), MEM(RCX,R13,1,64))
            VINSERTF32X4(ZMM(2), ZMM(2), MEM(RCX,RBX,4,64), IMM(1))
            VINSERTF32X4(ZMM(3), ZMM(3), MEM(RCX,R15,1,64), IMM(1))
            VINSERTF32X4(ZMM(4), ZMM(4), MEM(RCX,R13,2,64), IMM(1))
            VINSERTF32X4(ZMM(5), ZMM(5), MEM(RCX,R10,1,64), IMM(1))
            VFMADD213PS(ZMM(2), ZMM(1), ZMM(24))
            VFMADD213PS(ZMM(3), ZMM(1), ZMM(25))
            VFMADD213PS(ZMM(4), ZMM(1), ZMM(26))
            VFMADD213PS(ZMM(5), ZMM(1), ZMM(27))
            VMOVUPS(MEM(RCX,      64), YMM(2))
            VMOVUPS(MEM(RCX,RBX,1,64), YMM(3))
            VMOVUPS(MEM(RCX,RBX,2,64), YMM(4))
            VMOVUPS(MEM(RCX,R13,1,64), YMM(5))
            VEXTRACTF32X4(MEM(RCX,RBX,4,64), ZMM(2), IMM(1))
            VEXTRACTF32X4(MEM(RCX,R15,1,64), ZMM(3), IMM(1))
            VEXTRACTF32X4(MEM(RCX,R13,2,64), ZMM(4), IMM(1))
            VEXTRACTF32X4(MEM(RCX,R10,1,64), ZMM(5), IMM(1))
            VMOVUPS(YMM(2), MEM(RDX,      64))
            VMOVUPS(YMM(3), MEM(RDX,RBX,1,64))
            VMOVUPS(YMM(4), MEM(RDX,RBX,2,64))
            VMOVUPS(YMM(5), MEM(RDX,R13,1,64))
            VINSERTF32X4(ZMM(2), ZMM(2), MEM(RDX,RBX,4,64), IMM(1))
            VINSERTF32X4(ZMM(3), ZMM(3), MEM(RDX,R15,1,64), IMM(1))
            VINSERTF32X4(ZMM(4), ZMM(4), MEM(RDX,R13,2,64), IMM(1))
            VINSERTF32X4(ZMM(5), ZMM(5), MEM(RDX,R10,1,64), IMM(1))
            VFMADD213PS(ZMM(2), ZMM(1), ZMM(28))
            VFMADD213PS(ZMM(3), ZMM(1), ZMM(29))
            VFMADD213PS(ZMM(4), ZMM(1), ZMM(30))
            VFMADD213PS(ZMM(5), ZMM(1), ZMM(31))
            VMOVUPS(MEM(RDX,      64), YMM(2))
            VMOVUPS(MEM(RDX,RBX,1,64), YMM(3))
            VMOVUPS(MEM(RDX,RBX,2,64), YMM(4))
            VMOVUPS(MEM(RDX,R13,1,64), YMM(5))
            VEXTRACTF32X4(MEM(RDX,RBX,4,64), ZMM(2), IMM(1))
            VEXTRACTF32X4(MEM(RDX,R15,1,64), ZMM(3), IMM(1))
            VEXTRACTF32X4(MEM(RDX,R13,2,64), ZMM(4), IMM(1))
            VEXTRACTF32X4(MEM(RDX,R10,1,64), ZMM(5), IMM(1))

            // Now transpose the first 16 rows as a giant block

            VMULPD(ZMM( 8), ZMM( 8), ZMM(0))
            VMULPD(ZMM( 9), ZMM( 9), ZMM(0))
            VMULPD(ZMM(10), ZMM(10), ZMM(0))
            VMULPD(ZMM(11), ZMM(11), ZMM(0))
            VMULPD(ZMM(12), ZMM(12), ZMM(0))
            VMULPD(ZMM(13), ZMM(13), ZMM(0))
            VMULPD(ZMM(14), ZMM(14), ZMM(0))
            VMULPD(ZMM(15), ZMM(15), ZMM(0))
            VMULPD(ZMM(16), ZMM(16), ZMM(0))
            VMULPD(ZMM(17), ZMM(17), ZMM(0))
            VMULPD(ZMM(18), ZMM(18), ZMM(0))
            VMULPD(ZMM(19), ZMM(19), ZMM(0))
            VMULPD(ZMM(20), ZMM(20), ZMM(0))
            VMULPD(ZMM(21), ZMM(21), ZMM(0))
            VMULPD(ZMM(22), ZMM(22), ZMM(0))
            VMULPD(ZMM(23), ZMM(23), ZMM(0))
            /* 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F -  8 */
            /* 10 11 12 13 14 15 16 17 18 19 1A 1B 1C 1D 1E 1F -  9 */
            /* 20 21 22 23 24 25 26 27 28 29 2A 2B 2C 2D 2E 2F - 10 */
            /* 30 31 32 33 34 35 36 37 38 39 3A 3B 3C 3D 3E 3F - 11 */
            /* 40 41 42 43 44 45 46 47 48 49 4A 4B 4C 4D 4E 4F - 12 */
            /* 50 51 52 53 54 55 56 57 58 59 5A 5B 5C 5D 5E 5F - 13 */
            /* 60 61 62 63 64 65 66 67 68 69 6A 6B 6C 6D 6E 6F - 14 */
            /* 70 71 72 73 74 75 76 77 78 79 7A 7B 7C 7D 7E 7F - 15 */
            /* 80 81 82 83 84 85 86 87 88 89 8A 8B 8C 8D 8E 8F - 16 */
            /* 90 91 92 93 94 95 96 97 98 99 9A 9B 9C 9D 9E 9F - 17 */
            /* A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 AA AB AC AD AE AF - 18 */
            /* B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 BA BB BC BD BE BF - 19 */
            /* C0 C1 C2 C3 C4 C5 C6 C7 C8 C9 CA CB CC CD CE CF - 20 */
            /* D0 D1 D2 D3 D4 D5 D6 D7 D8 D9 DA DB DC DD DE DF - 21 */
            /* E0 E1 E2 E3 E4 EA E6 E7 E8 E9 EA EB EC ED EE EF - 22 */
            /* F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 FA FB FC FD FE FF - 23 */
            VUNPCKLPS(ZMM(24), ZMM( 8), ZMM( 9))
            VUNPCKHPS(ZMM(25), ZMM(10), ZMM(11))
            VUNPCKHPS(ZMM(26), ZMM(12), ZMM(13))
            VUNPCKLPS(ZMM(27), ZMM(14), ZMM(15))
            VUNPCKLPS(ZMM(28), ZMM(16), ZMM(17))
            VUNPCKHPS(ZMM(29), ZMM(18), ZMM(19))
            VUNPCKHPS(ZMM(30), ZMM(20), ZMM(21))
            VUNPCKLPS(ZMM(31), ZMM(22), ZMM(23))
            VUNPCKHPS(ZMM( 9), ZMM( 8), ZMM( 9))
            VUNPCKLPS(ZMM(10), ZMM(10), ZMM(11))
            VUNPCKLPS(ZMM(12), ZMM(12), ZMM(13))
            VUNPCKHPS(ZMM(15), ZMM(14), ZMM(15))
            VUNPCKHPS(ZMM(17), ZMM(16), ZMM(17))
            VUNPCKLPS(ZMM(18), ZMM(18), ZMM(19))
            VUNPCKLPS(ZMM(20), ZMM(20), ZMM(21))
            VUNPCKHPS(ZMM(23), ZMM(22), ZMM(23))
            /* 00 10 02 12 04 14 06 16 08 18 0A 1A 0C 1C 0E 1E - 24 */
            /* 01 11 03 13 05 15 07 17 09 19 0B 1B 0D 1D 0F 1F -  9 */
            /* 20 30 22 32 24 34 26 36 28 38 2A 3A 2C 3C 2E 3E - 10 */
            /* 21 31 23 33 25 35 27 37 29 39 2B 3B 2D 3D 2F 3F - 25 */
            /* 40 50 42 52 44 54 46 56 48 58 4A 5A 4C 5C 4E 5E - 12 */
            /* 41 51 43 53 45 55 47 57 49 59 4B 5B 4D 5D 4F 5F - 26 */
            /* 60 70 62 72 64 74 66 76 68 78 6A 7A 6C 7C 6E 7E - 27 */
            /* 61 71 63 73 65 75 67 77 69 79 6B 7B 6D 7D 6F 7F - 15 */
            /* 80 90 82 92 84 94 86 96 88 98 8A 8A 8C 9C 8E 9E - 28 */
            /* 81 91 83 93 85 95 87 97 89 99 8B 9B 8D 9D 8F 9F - 17 */
            /* A0 B0 A2 B2 A4 B4 A6 B6 A8 B8 AA BA AC BC AE BE - 18 */
            /* A1 B1 A3 B3 A5 B5 A7 B7 A9 B9 AB BB AD BD AF BF - 29 */
            /* C0 D0 C2 D2 C4 D4 C6 D6 C8 D8 CA DA CC DC CE DE - 20 */
            /* C1 D1 C3 D3 C5 D5 C7 D7 C9 D9 CB DB CD DD CF DF - 30 */
            /* E0 F0 E2 F2 E4 F4 E6 F6 E8 F8 EA FA EC FC EE FE - 31 */
            /* E1 F1 E3 F3 E5 F5 E7 F7 E9 F9 EB FB ED FD EF FF - 23 */
            VSHUFPS(ZMM( 8), ZMM(24), ZMM(10), IMM(0x44))
            VSHUFPS(ZMM(11), ZMM( 9), ZMM(25), IMM(0xee))
            VSHUFPS(ZMM(14), ZMM(12), ZMM(27), IMM(0xee))
            VSHUFPS(ZMM(13), ZMM(26), ZMM(15), IMM(0x44))
            VSHUFPS(ZMM(16), ZMM(28), ZMM(18), IMM(0x44))
            VSHUFPS(ZMM(19), ZMM(17), ZMM(29), IMM(0xee))
            VSHUFPS(ZMM(22), ZMM(20), ZMM(31), IMM(0xee))
            VSHUFPS(ZMM(21), ZMM(30), ZMM(23), IMM(0x44))
            VSHUFPS(ZMM(10), ZMM(24), ZMM(10), IMM(0xee))
            VSHUFPS(ZMM( 9), ZMM( 9), ZMM(25), IMM(0x44))
            VSHUFPS(ZMM(12), ZMM(12), ZMM(27), IMM(0x44))
            VSHUFPS(ZMM(15), ZMM(26), ZMM(15), IMM(0xee))
            VSHUFPS(ZMM(18), ZMM(28), ZMM(18), IMM(0xee))
            VSHUFPS(ZMM(17), ZMM(17), ZMM(29), IMM(0x44))
            VSHUFPS(ZMM(20), ZMM(20), ZMM(31), IMM(0x44))
            VSHUFPS(ZMM(23), ZMM(30), ZMM(23), IMM(0xee))
            /* 00 10 20 30 04 14 24 34 08 18 28 38 0C 1C 2C 3C -  8 */
            /* 01 11 03 31 05 15 25 35 09 19 29 39 0D 1D 2D 3D -  9 */
            /* 02 12 22 32 06 16 26 36 0A 1A 2A 3A 0E 1E 2E 3E - 10 */
            /* 03 13 23 33 07 17 27 37 0B 1B 2B 3B 0F 1F 2F 3F - 11 */
            /* 40 50 60 70 44 54 64 74 48 58 68 78 4C 5C 6C 7C - 12 */
            /* 41 51 61 71 45 55 65 75 49 59 69 79 4D 5D 6D 7D - 13 */
            /* 42 52 62 72 46 56 66 76 4A 5A 6A 7A 4E 5E 6E 7E - 14 */
            /* 43 53 63 73 47 57 67 77 4B 5B 6B 7B 4F 5F 6F 7F - 15 */
            /* 80 90 A0 B0 84 94 A4 B4 88 98 A8 B8 8C 9C AC BC - 16 */
            /* 81 91 A1 B1 85 95 A5 B5 89 99 A9 B9 8D 9D AD BD - 17 */
            /* 82 92 A2 B2 86 96 A6 B6 8A 9A AA BA 8E 9E AE BE - 18 */
            /* 83 93 A3 B3 87 97 A7 B7 8B 9B AB BB 8F 9F AF BF - 19 */
            /* C0 D0 E0 F0 C4 D4 E4 F4 C8 D8 E8 F8 CC DC EC FC - 20 */
            /* C1 D1 E1 F1 C5 D5 E5 F5 C9 D9 E9 F9 CD DD ED FD - 21 */
            /* C2 D2 E2 F2 C6 D6 E6 F6 CA DA EA FA CE DE EE FE - 22 */
            /* C3 D3 E3 F3 C7 D7 E7 F7 CB DB EB FB CF DF EF FF - 23 */
            VSHUFF32X4(ZMM(24), ZMM( 8), ZMM(12), IMM(0x8d))
            VBLENDMPS(ZMM( 8) MASK_K(1), ZMM(24), ZMM( 8))
            VBLENDMPS(ZMM(12) MASK_K(1), ZMM(12), ZMM(24))
            VSHUFF32X4(ZMM(25), ZMM( 9), ZMM(13), IMM(0x8d))
            VBLENDMPS(ZMM( 9) MASK_K(1), ZMM(25), ZMM( 9))
            VBLENDMPS(ZMM(13) MASK_K(1), ZMM(13), ZMM(25))
            VSHUFF32X4(ZMM(26), ZMM(10), ZMM(14), IMM(0x8d))
            VBLENDMPS(ZMM(10) MASK_K(1), ZMM(26), ZMM(10))
            VBLENDMPS(ZMM(14) MASK_K(1), ZMM(14), ZMM(26))
            VSHUFF32X4(ZMM(27), ZMM(11), ZMM(15), IMM(0x8d))
            VBLENDMPS(ZMM(11) MASK_K(1), ZMM(27), ZMM(11))
            VBLENDMPS(ZMM(15) MASK_K(1), ZMM(15), ZMM(27))
            VSHUFF32X4(ZMM(28), ZMM(16), ZMM(20), IMM(0x8d))
            VBLENDMPS(ZMM(16) MASK_K(1), ZMM(28), ZMM(16))
            VBLENDMPS(ZMM(20) MASK_K(1), ZMM(20), ZMM(28))
            VSHUFF32X4(ZMM(29), ZMM(17), ZMM(21), IMM(0x8d))
            VBLENDMPS(ZMM(17) MASK_K(1), ZMM(29), ZMM(17))
            VBLENDMPS(ZMM(21) MASK_K(1), ZMM(21), ZMM(29))
            VSHUFF32X4(ZMM(30), ZMM(18), ZMM(22), IMM(0x8d))
            VBLENDMPS(ZMM(18) MASK_K(1), ZMM(30), ZMM(18))
            VBLENDMPS(ZMM(22) MASK_K(1), ZMM(22), ZMM(30))
            VSHUFF32X4(ZMM(31), ZMM(19), ZMM(23), IMM(0x8d))
            VBLENDMPS(ZMM(19) MASK_K(1), ZMM(31), ZMM(19))
            VBLENDMPS(ZMM(23) MASK_K(1), ZMM(23), ZMM(31))
            /* 00 10 20 30 40 50 60 70 08 18 28 38 48 58 68 78 -  8 */
            /* 01 11 03 31 41 51 61 71 09 19 29 39 49 59 69 79 -  9 */
            /* 02 12 22 32 42 52 62 72 0A 1A 2A 3A 4A 5A 6A 7A - 10 */
            /* 03 13 23 33 43 53 63 73 0B 1B 2B 3B 4B 5B 6B 7B - 11 */
            /* 04 14 24 34 44 54 64 74 0C 1C 2C 3C 4C 5C 6C 7C - 12 */
            /* 05 15 25 35 45 55 65 75 0D 1D 2D 3D 4D 5D 6D 7D - 13 */
            /* 06 16 26 36 46 56 66 76 0E 1E 2E 3E 4E 5E 6E 7E - 14 */
            /* 07 17 27 37 47 57 67 77 0F 1F 2F 3F 4F 5F 6F 7F - 15 */
            /* 80 90 A0 B0 C0 D0 E0 F0 88 98 A8 B8 C8 D8 E8 F8 - 16 */
            /* 81 91 A1 B1 C1 D1 E1 F1 85 95 A5 B5 89 99 A9 B9 - 17 */
            /* 82 92 A2 B2 C2 D2 E2 F2 8A 9A AA BA CA DA EA FA - 18 */
            /* 83 93 A3 B3 C3 D3 E3 F3 8B 9B AB BB CB DB EB FB - 19 */
            /* 84 94 A4 B4 C4 D4 E4 F4 8C 9C AC BC CC DC EC FC - 20 */
            /* C5 D5 E5 F5 C9 D9 E9 F9 8D 9D AD BD CD DD ED FD - 21 */
            /* 86 96 A6 B6 C6 D6 E6 F6 8E 9E AE BE CE DE EE FE - 22 */
            /* 87 97 A7 B7 C7 D7 E7 F7 8F 9F AF BF CF DF EF FF - 23 */
            VSHUFF32X4(ZMM(24), ZMM( 8), ZMM(16), IMM(0x4e))
            VBLENDMPS(ZMM( 8) MASK_K(2), ZMM(24), ZMM( 8))
            VBLENDMPS(ZMM(16) MASK_K(2), ZMM(16), ZMM(24))
            VSHUFF32X4(ZMM(25), ZMM( 9), ZMM(17), IMM(0x4e))
            VBLENDMPS(ZMM( 9) MASK_K(2), ZMM(25), ZMM( 9))
            VBLENDMPS(ZMM(17) MASK_K(2), ZMM(17), ZMM(25))
            VSHUFF32X4(ZMM(26), ZMM(10), ZMM(18), IMM(0x4e))
            VBLENDMPS(ZMM(10) MASK_K(2), ZMM(26), ZMM(10))
            VBLENDMPS(ZMM(18) MASK_K(2), ZMM(18), ZMM(26))
            VSHUFF32X4(ZMM(27), ZMM(11), ZMM(19), IMM(0x4e))
            VBLENDMPS(ZMM(11) MASK_K(2), ZMM(27), ZMM(11))
            VBLENDMPS(ZMM(19) MASK_K(2), ZMM(19), ZMM(27))
            VSHUFF32X4(ZMM(28), ZMM(12), ZMM(20), IMM(0x4e))
            VBLENDMPS(ZMM(12) MASK_K(2), ZMM(28), ZMM(12))
            VBLENDMPS(ZMM(20) MASK_K(2), ZMM(20), ZMM(28))
            VSHUFF32X4(ZMM(29), ZMM(13), ZMM(21), IMM(0x4e))
            VBLENDMPS(ZMM(13) MASK_K(2), ZMM(29), ZMM(13))
            VBLENDMPS(ZMM(21) MASK_K(2), ZMM(21), ZMM(29))
            VSHUFF32X4(ZMM(30), ZMM(14), ZMM(22), IMM(0x4e))
            VBLENDMPS(ZMM(14) MASK_K(2), ZMM(30), ZMM(14))
            VBLENDMPS(ZMM(22) MASK_K(2), ZMM(22), ZMM(30))
            VSHUFF32X4(ZMM(31), ZMM(15), ZMM(23), IMM(0x4e))
            VBLENDMPS(ZMM(15) MASK_K(2), ZMM(31), ZMM(15))
            VBLENDMPS(ZMM(23) MASK_K(2), ZMM(23), ZMM(31))
            /* 00 10 20 30 40 50 60 70 80 90 A0 B0 C0 D0 E0 F0 -  8 */
            /* 01 11 03 31 41 51 61 71 81 91 A1 B1 C1 D1 E1 F1 -  9 */
            /* 02 12 22 32 42 52 62 72 82 92 A2 B2 C2 D2 E2 F2 - 10 */
            /* 03 13 23 33 43 53 63 73 83 93 A3 B3 C3 D3 E3 F3 - 11 */
            /* 04 14 24 34 44 54 64 74 84 94 A4 B4 C4 D4 E4 F4 - 12 */
            /* 05 15 25 35 45 55 65 75 85 95 A5 B5 C5 D5 E5 F5 - 13 */
            /* 06 16 26 36 46 56 66 76 86 96 A6 B6 C6 D6 E6 F6 - 14 */
            /* 07 17 27 37 47 57 67 77 87 97 A7 B7 C7 D7 E7 F7 - 15 */
            /* 08 18 28 38 48 58 68 78 88 98 A8 B8 C8 D8 E8 F8 - 16 */
            /* 09 19 09 39 49 59 69 79 89 99 A9 B9 C9 D9 E9 F9 - 17 */
            /* 0A 1A 2A 3A 4A 5A 6A 7A 8A 9A AA BA CA DA EA FA - 18 */
            /* 0B 1B 2B 3B 4B 5B 6B 7B 8B 9B AB BB CB DB EB FB - 19 */
            /* 0C 1C 2C 3C 4C 5C 6C 7C 8C 9C AC BC CC DC EC FC - 20 */
            /* 0D 1D 2D 3D 4D 5D 6D 7D 8D 9D AD BD CD DD ED FD - 21 */
            /* 0E 1E 2E 3E 4E 5E 6E 7E 8E 9E AE BE CE DE EE FE - 22 */
            /* 0F 1F 2F 3F 4F 5F 6F 7F 8F 9F AF BF CF DF EF FF - 23 */
            VFMADD231PS(ZMM( 8), ZMM(1), MEM(RCX      ))
            VMOVUPS(MEM(RCX      ), ZMM( 8))
            VFMADD231PS(ZMM( 9), ZMM(1), MEM(RCX,RBX,1))
            VMOVUPS(MEM(RCX,RBX,1), ZMM( 9))
            VFMADD231PS(ZMM(10), ZMM(1), MEM(RCX,RBX,2))
            VMOVUPS(MEM(RCX,RBX,2), ZMM(10))
            VFMADD231PS(ZMM(11), ZMM(1), MEM(RCX,R13,1))
            VMOVUPS(MEM(RCX,R13,1), ZMM(11))
            VFMADD231PS(ZMM(12), ZMM(1), MEM(RCX,RBX,4))
            VMOVUPS(MEM(RCX,RBX,4), ZMM(12))
            VFMADD231PS(ZMM(13), ZMM(1), MEM(RCX,R15,1))
            VMOVUPS(MEM(RCX,R15,1), ZMM(13))
            VFMADD231PS(ZMM(14), ZMM(1), MEM(RCX,R13,2))
            VMOVUPS(MEM(RCX,R13,2), ZMM(14))
            VFMADD231PS(ZMM(15), ZMM(1), MEM(RCX,R10,1))
            VMOVUPS(MEM(RCX,R10,1), ZMM(15))
            VFMADD231PS(ZMM(16), ZMM(1), MEM(RDX      ))
            VMOVUPS(MEM(RDX      ), ZMM(16))
            VFMADD231PS(ZMM(17), ZMM(1), MEM(RDX,RBX,1))
            VMOVUPS(MEM(RDX,RBX,1), ZMM(17))
            VFMADD231PS(ZMM(18), ZMM(1), MEM(RDX,RBX,2))
            VMOVUPS(MEM(RDX,RBX,2), ZMM(18))
            VFMADD231PS(ZMM(19), ZMM(1), MEM(RDX,R13,1))
            VMOVUPS(MEM(RDX,R13,1), ZMM(19))
            VFMADD231PS(ZMM(20), ZMM(1), MEM(RDX,RBX,4))
            VMOVUPS(MEM(RDX,RBX,4), ZMM(20))
            VFMADD231PS(ZMM(21), ZMM(1), MEM(RDX,R15,1))
            VMOVUPS(MEM(RDX,R15,1), ZMM(21))
            VFMADD231PS(ZMM(22), ZMM(1), MEM(RDX,R13,2))
            VMOVUPS(MEM(RDX,R13,2), ZMM(22))
            VFMADD231PS(ZMM(23), ZMM(1), MEM(RDX,R10,1))
            VMOVUPS(MEM(RDX,R10,1), ZMM(23))

        JMP(END)
        LABEL(COLSTORBZ)

            // Transpose and write out the last 8 rows so we can use them as temporaries

            VMULPD(ZMM(24), ZMM(24), ZMM(0))
            VMULPD(ZMM(25), ZMM(25), ZMM(0))
            VMULPD(ZMM(26), ZMM(26), ZMM(0))
            VMULPD(ZMM(27), ZMM(27), ZMM(0))
            VMULPD(ZMM(28), ZMM(28), ZMM(0))
            VMULPD(ZMM(29), ZMM(29), ZMM(0))
            VMULPD(ZMM(30), ZMM(30), ZMM(0))
            VMULPD(ZMM(31), ZMM(31), ZMM(0))
            /* 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F - 24 */
            /* 10 11 12 13 14 15 16 17 18 19 1A 1B 1C 1D 1E 1F - 25 */
            /* 20 21 22 23 24 25 26 27 28 29 2A 2B 2C 2D 2E 2F - 26 */
            /* 30 31 32 33 34 35 36 37 38 39 3A 3B 3C 3D 3E 3F - 27 */
            /* 40 41 42 43 44 45 46 47 48 49 4A 4B 4C 4D 4E 4F - 28 */
            /* 50 51 52 53 54 55 56 57 58 59 5A 5B 5C 5D 5E 5F - 29 */
            /* 60 61 62 63 64 65 66 67 68 69 6A 6B 6C 6D 6E 6F - 30 */
            /* 70 71 72 73 74 75 76 77 78 79 7A 7B 7C 7D 7E 7F - 31 */
            VUNPCKLPS(ZMM( 2), ZMM(24), ZMM(25))
            VUNPCKHPS(ZMM( 3), ZMM(26), ZMM(27))
            VUNPCKLPS(ZMM( 4), ZMM(28), ZMM(29))
            VUNPCKHPS(ZMM( 5), ZMM(30), ZMM(31))
            VUNPCKHPS(ZMM(25), ZMM(24), ZMM(25))
            VUNPCKLPS(ZMM(26), ZMM(26), ZMM(27))
            VUNPCKHPS(ZMM(29), ZMM(28), ZMM(29))
            VUNPCKLPS(ZMM(30), ZMM(30), ZMM(31))
            /* 00 10 02 12 04 14 06 16 08 18 0A 1A 0C 1C 0E 1E -  2 */
            /* 01 11 03 13 05 15 07 17 09 19 0B 1B 0D 1D 0F 1F - 25 */
            /* 20 30 22 32 24 34 26 36 28 38 2A 3A 2C 3C 2E 3E - 26 */
            /* 21 31 23 33 25 35 27 37 29 39 2B 3B 2D 3D 2F 3F -  3 */
            /* 40 50 42 52 44 54 46 56 48 58 4A 5A 4C 5C 4E 5E -  4 */
            /* 41 51 43 53 45 55 47 57 49 59 4B 5B 4D 5D 4F 5F - 29 */
            /* 60 70 62 72 64 74 66 76 68 78 6A 7A 6C 7C 6E 7E - 30 */
            /* 61 71 63 73 65 75 67 77 69 79 6B 7B 6D 7D 6F 7F -  5 */
            VSHUFPS(ZMM(24), ZMM( 2), ZMM(26), IMM(0x44))
            VSHUFPS(ZMM(27), ZMM(25), ZMM( 3), IMM(0xee))
            VSHUFPS(ZMM(28), ZMM( 4), ZMM(30), IMM(0x44))
            VSHUFPS(ZMM(31), ZMM(29), ZMM( 5), IMM(0xee))
            VSHUFPS(ZMM(26), ZMM( 2), ZMM(26), IMM(0xee))
            VSHUFPS(ZMM(25), ZMM(25), ZMM( 3), IMM(0x44))
            VSHUFPS(ZMM(30), ZMM( 4), ZMM(30), IMM(0xee))
            VSHUFPS(ZMM(29), ZMM(29), ZMM( 5), IMM(0x44))
            /* 00 10 20 30 04 14 24 34 08 18 28 38 0C 1C 2C 3C - 24 */
            /* 01 11 03 31 05 15 25 35 09 19 29 39 0D 1D 2D 3D - 25 */
            /* 02 12 22 32 06 16 26 36 0A 1A 2A 3A 0E 1E 2E 3E - 26 */
            /* 03 13 23 33 07 17 27 37 0B 1B 2B 3B 0F 1F 2F 3F - 27 */
            /* 40 50 60 70 44 54 64 74 48 58 68 78 4C 5C 6C 7C - 28 */
            /* 41 51 61 71 45 55 65 75 49 59 69 79 4D 5D 6D 7D - 29 */
            /* 42 52 62 72 46 56 66 76 4A 5A 6A 7A 4E 5E 6E 7E - 30 */
            /* 43 53 63 73 47 57 67 77 4B 5B 6B 7B 4F 5F 6F 7F - 31 */
            VBLENDMPS(ZMM(2) MASK_K(1), ZMM(24), ZMM(28))
            VPERMPD(ZMM(2), ZMM(2), IMM(0x4e))
            VBLENDMPS(ZMM(24) MASK_K(1), ZMM(2), ZMM(24))
            VBLENDMPS(ZMM(28) MASK_K(1), ZMM(28), ZMM(2))
            VBLENDMPS(ZMM(3) MASK_K(1), ZMM(25), ZMM(29))
            VPERMPD(ZMM(3), ZMM(3), IMM(0x4e))
            VBLENDMPS(ZMM(25) MASK_K(1), ZMM(3), ZMM(25))
            VBLENDMPS(ZMM(29) MASK_K(1), ZMM(29), ZMM(3))
            VBLENDMPS(ZMM(4) MASK_K(1), ZMM(26), ZMM(30))
            VPERMPD(ZMM(4), ZMM(4), IMM(0x4e))
            VBLENDMPS(ZMM(26) MASK_K(1), ZMM(4), ZMM(26))
            VBLENDMPS(ZMM(30) MASK_K(1), ZMM(30), ZMM(4))
            VBLENDMPS(ZMM(5) MASK_K(1), ZMM(27), ZMM(31))
            VPERMPD(ZMM(5), ZMM(5), IMM(0x4e))
            VBLENDMPS(ZMM(27) MASK_K(1), ZMM(5), ZMM(27))
            VBLENDMPS(ZMM(31) MASK_K(1), ZMM(31), ZMM(5))
            /* 00 10 20 30 40 50 60 70 04 14 24 34 44 54 64 74 - 24 */
            /* 01 11 03 31 41 51 61 71 05 15 25 35 45 55 65 75 - 25 */
            /* 02 12 22 32 42 52 62 72 06 16 26 36 46 56 66 76 - 26 */
            /* 03 13 23 33 43 53 63 73 07 17 27 37 47 57 67 77 - 27 */
            /* 08 18 28 38 48 58 68 78 0C 1C 2C 3C 4C 5C 6C 7C - 28 */
            /* 09 19 29 39 49 59 69 79 0D 1D 2D 3D 4D 5D 6D 7D - 29 */
            /* 0A 1A 2A 3A 4A 5A 6A 7A 0E 1E 2E 3E 4E 5E 6E 7E - 30 */
            /* 0B 1B 2B 3B 4B 5B 6B 7B 0F 1F 2F 3F 4F 5F 6F 7F - 31 */
            VMOVUPS(MEM(RCX,      64), YMM(2))
            VMOVUPS(MEM(RCX,RBX,1,64), YMM(3))
            VMOVUPS(MEM(RCX,RBX,2,64), YMM(4))
            VMOVUPS(MEM(RCX,R13,1,64), YMM(5))
            VEXTRACTF32X4(MEM(RCX,RBX,4,64), ZMM(2), IMM(1))
            VEXTRACTF32X4(MEM(RCX,R15,1,64), ZMM(3), IMM(1))
            VEXTRACTF32X4(MEM(RCX,R13,2,64), ZMM(4), IMM(1))
            VEXTRACTF32X4(MEM(RCX,R10,1,64), ZMM(5), IMM(1))
            VMOVUPS(MEM(RDX,      64), YMM(2))
            VMOVUPS(MEM(RDX,RBX,1,64), YMM(3))
            VMOVUPS(MEM(RDX,RBX,2,64), YMM(4))
            VMOVUPS(MEM(RDX,R13,1,64), YMM(5))
            VEXTRACTF32X4(MEM(RDX,RBX,4,64), ZMM(2), IMM(1))
            VEXTRACTF32X4(MEM(RDX,R15,1,64), ZMM(3), IMM(1))
            VEXTRACTF32X4(MEM(RDX,R13,2,64), ZMM(4), IMM(1))
            VEXTRACTF32X4(MEM(RDX,R10,1,64), ZMM(5), IMM(1))

            // Now transpose the first 16 rows as a giant block

            VMULPD(ZMM( 8), ZMM( 8), ZMM(0))
            VMULPD(ZMM( 9), ZMM( 9), ZMM(0))
            VMULPD(ZMM(10), ZMM(10), ZMM(0))
            VMULPD(ZMM(11), ZMM(11), ZMM(0))
            VMULPD(ZMM(12), ZMM(12), ZMM(0))
            VMULPD(ZMM(13), ZMM(13), ZMM(0))
            VMULPD(ZMM(14), ZMM(14), ZMM(0))
            VMULPD(ZMM(15), ZMM(15), ZMM(0))
            VMULPD(ZMM(16), ZMM(16), ZMM(0))
            VMULPD(ZMM(17), ZMM(17), ZMM(0))
            VMULPD(ZMM(18), ZMM(18), ZMM(0))
            VMULPD(ZMM(19), ZMM(19), ZMM(0))
            VMULPD(ZMM(20), ZMM(20), ZMM(0))
            VMULPD(ZMM(21), ZMM(21), ZMM(0))
            VMULPD(ZMM(22), ZMM(22), ZMM(0))
            VMULPD(ZMM(23), ZMM(23), ZMM(0))
            /* 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F -  8 */
            /* 10 11 12 13 14 15 16 17 18 19 1A 1B 1C 1D 1E 1F -  9 */
            /* 20 21 22 23 24 25 26 27 28 29 2A 2B 2C 2D 2E 2F - 10 */
            /* 30 31 32 33 34 35 36 37 38 39 3A 3B 3C 3D 3E 3F - 11 */
            /* 40 41 42 43 44 45 46 47 48 49 4A 4B 4C 4D 4E 4F - 12 */
            /* 50 51 52 53 54 55 56 57 58 59 5A 5B 5C 5D 5E 5F - 13 */
            /* 60 61 62 63 64 65 66 67 68 69 6A 6B 6C 6D 6E 6F - 14 */
            /* 70 71 72 73 74 75 76 77 78 79 7A 7B 7C 7D 7E 7F - 15 */
            /* 80 81 82 83 84 85 86 87 88 89 8A 8B 8C 8D 8E 8F - 16 */
            /* 90 91 92 93 94 95 96 97 98 99 9A 9B 9C 9D 9E 9F - 17 */
            /* A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 AA AB AC AD AE AF - 18 */
            /* B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 BA BB BC BD BE BF - 19 */
            /* C0 C1 C2 C3 C4 C5 C6 C7 C8 C9 CA CB CC CD CE CF - 20 */
            /* D0 D1 D2 D3 D4 D5 D6 D7 D8 D9 DA DB DC DD DE DF - 21 */
            /* E0 E1 E2 E3 E4 EA E6 E7 E8 E9 EA EB EC ED EE EF - 22 */
            /* F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 FA FB FC FD FE FF - 23 */
            VUNPCKLPS(ZMM(24), ZMM( 8), ZMM( 9))
            VUNPCKHPS(ZMM(25), ZMM(10), ZMM(11))
            VUNPCKHPS(ZMM(26), ZMM(12), ZMM(13))
            VUNPCKLPS(ZMM(27), ZMM(14), ZMM(15))
            VUNPCKLPS(ZMM(28), ZMM(16), ZMM(17))
            VUNPCKHPS(ZMM(29), ZMM(18), ZMM(19))
            VUNPCKHPS(ZMM(30), ZMM(20), ZMM(21))
            VUNPCKLPS(ZMM(31), ZMM(22), ZMM(23))
            VUNPCKHPS(ZMM( 9), ZMM( 8), ZMM( 9))
            VUNPCKLPS(ZMM(10), ZMM(10), ZMM(11))
            VUNPCKLPS(ZMM(12), ZMM(12), ZMM(13))
            VUNPCKHPS(ZMM(15), ZMM(14), ZMM(15))
            VUNPCKHPS(ZMM(17), ZMM(16), ZMM(17))
            VUNPCKLPS(ZMM(18), ZMM(18), ZMM(19))
            VUNPCKLPS(ZMM(20), ZMM(20), ZMM(21))
            VUNPCKHPS(ZMM(23), ZMM(22), ZMM(23))
            /* 00 10 02 12 04 14 06 16 08 18 0A 1A 0C 1C 0E 1E - 24 */
            /* 01 11 03 13 05 15 07 17 09 19 0B 1B 0D 1D 0F 1F -  9 */
            /* 20 30 22 32 24 34 26 36 28 38 2A 3A 2C 3C 2E 3E - 10 */
            /* 21 31 23 33 25 35 27 37 29 39 2B 3B 2D 3D 2F 3F - 25 */
            /* 40 50 42 52 44 54 46 56 48 58 4A 5A 4C 5C 4E 5E - 12 */
            /* 41 51 43 53 45 55 47 57 49 59 4B 5B 4D 5D 4F 5F - 26 */
            /* 60 70 62 72 64 74 66 76 68 78 6A 7A 6C 7C 6E 7E - 27 */
            /* 61 71 63 73 65 75 67 77 69 79 6B 7B 6D 7D 6F 7F - 15 */
            /* 80 90 82 92 84 94 86 96 88 98 8A 8A 8C 9C 8E 9E - 28 */
            /* 81 91 83 93 85 95 87 97 89 99 8B 9B 8D 9D 8F 9F - 17 */
            /* A0 B0 A2 B2 A4 B4 A6 B6 A8 B8 AA BA AC BC AE BE - 18 */
            /* A1 B1 A3 B3 A5 B5 A7 B7 A9 B9 AB BB AD BD AF BF - 29 */
            /* C0 D0 C2 D2 C4 D4 C6 D6 C8 D8 CA DA CC DC CE DE - 20 */
            /* C1 D1 C3 D3 C5 D5 C7 D7 C9 D9 CB DB CD DD CF DF - 30 */
            /* E0 F0 E2 F2 E4 F4 E6 F6 E8 F8 EA FA EC FC EE FE - 31 */
            /* E1 F1 E3 F3 E5 F5 E7 F7 E9 F9 EB FB ED FD EF FF - 23 */
            VSHUFPS(ZMM( 8), ZMM(24), ZMM(10), IMM(0x44))
            VSHUFPS(ZMM(11), ZMM( 9), ZMM(25), IMM(0xee))
            VSHUFPS(ZMM(14), ZMM(12), ZMM(27), IMM(0xee))
            VSHUFPS(ZMM(13), ZMM(26), ZMM(15), IMM(0x44))
            VSHUFPS(ZMM(16), ZMM(28), ZMM(18), IMM(0x44))
            VSHUFPS(ZMM(19), ZMM(17), ZMM(29), IMM(0xee))
            VSHUFPS(ZMM(22), ZMM(20), ZMM(31), IMM(0xee))
            VSHUFPS(ZMM(21), ZMM(30), ZMM(23), IMM(0x44))
            VSHUFPS(ZMM(10), ZMM(24), ZMM(10), IMM(0xee))
            VSHUFPS(ZMM( 9), ZMM( 9), ZMM(25), IMM(0x44))
            VSHUFPS(ZMM(12), ZMM(12), ZMM(27), IMM(0x44))
            VSHUFPS(ZMM(15), ZMM(26), ZMM(15), IMM(0xee))
            VSHUFPS(ZMM(18), ZMM(28), ZMM(18), IMM(0xee))
            VSHUFPS(ZMM(17), ZMM(17), ZMM(29), IMM(0x44))
            VSHUFPS(ZMM(20), ZMM(20), ZMM(31), IMM(0x44))
            VSHUFPS(ZMM(23), ZMM(30), ZMM(23), IMM(0xee))
            /* 00 10 20 30 04 14 24 34 08 18 28 38 0C 1C 2C 3C -  8 */
            /* 01 11 03 31 05 15 25 35 09 19 29 39 0D 1D 2D 3D -  9 */
            /* 02 12 22 32 06 16 26 36 0A 1A 2A 3A 0E 1E 2E 3E - 10 */
            /* 03 13 23 33 07 17 27 37 0B 1B 2B 3B 0F 1F 2F 3F - 11 */
            /* 40 50 60 70 44 54 64 74 48 58 68 78 4C 5C 6C 7C - 12 */
            /* 41 51 61 71 45 55 65 75 49 59 69 79 4D 5D 6D 7D - 13 */
            /* 42 52 62 72 46 56 66 76 4A 5A 6A 7A 4E 5E 6E 7E - 14 */
            /* 43 53 63 73 47 57 67 77 4B 5B 6B 7B 4F 5F 6F 7F - 15 */
            /* 80 90 A0 B0 84 94 A4 B4 88 98 A8 B8 8C 9C AC BC - 16 */
            /* 81 91 A1 B1 85 95 A5 B5 89 99 A9 B9 8D 9D AD BD - 17 */
            /* 82 92 A2 B2 86 96 A6 B6 8A 9A AA BA 8E 9E AE BE - 18 */
            /* 83 93 A3 B3 87 97 A7 B7 8B 9B AB BB 8F 9F AF BF - 19 */
            /* C0 D0 E0 F0 C4 D4 E4 F4 C8 D8 E8 F8 CC DC EC FC - 20 */
            /* C1 D1 E1 F1 C5 D5 E5 F5 C9 D9 E9 F9 CD DD ED FD - 21 */
            /* C2 D2 E2 F2 C6 D6 E6 F6 CA DA EA FA CE DE EE FE - 22 */
            /* C3 D3 E3 F3 C7 D7 E7 F7 CB DB EB FB CF DF EF FF - 23 */
            VSHUFF32X4(ZMM(24), ZMM( 8), ZMM(12), IMM(0x8d))
            VBLENDMPS(ZMM( 8) MASK_K(1), ZMM(24), ZMM( 8))
            VBLENDMPS(ZMM(12) MASK_K(1), ZMM(12), ZMM(24))
            VSHUFF32X4(ZMM(25), ZMM( 9), ZMM(13), IMM(0x8d))
            VBLENDMPS(ZMM( 9) MASK_K(1), ZMM(25), ZMM( 9))
            VBLENDMPS(ZMM(13) MASK_K(1), ZMM(13), ZMM(25))
            VSHUFF32X4(ZMM(26), ZMM(10), ZMM(14), IMM(0x8d))
            VBLENDMPS(ZMM(10) MASK_K(1), ZMM(26), ZMM(10))
            VBLENDMPS(ZMM(14) MASK_K(1), ZMM(14), ZMM(26))
            VSHUFF32X4(ZMM(27), ZMM(11), ZMM(15), IMM(0x8d))
            VBLENDMPS(ZMM(11) MASK_K(1), ZMM(27), ZMM(11))
            VBLENDMPS(ZMM(15) MASK_K(1), ZMM(15), ZMM(27))
            VSHUFF32X4(ZMM(28), ZMM(16), ZMM(20), IMM(0x8d))
            VBLENDMPS(ZMM(16) MASK_K(1), ZMM(28), ZMM(16))
            VBLENDMPS(ZMM(20) MASK_K(1), ZMM(20), ZMM(28))
            VSHUFF32X4(ZMM(29), ZMM(17), ZMM(21), IMM(0x8d))
            VBLENDMPS(ZMM(17) MASK_K(1), ZMM(29), ZMM(17))
            VBLENDMPS(ZMM(21) MASK_K(1), ZMM(21), ZMM(29))
            VSHUFF32X4(ZMM(30), ZMM(18), ZMM(22), IMM(0x8d))
            VBLENDMPS(ZMM(18) MASK_K(1), ZMM(30), ZMM(18))
            VBLENDMPS(ZMM(22) MASK_K(1), ZMM(22), ZMM(30))
            VSHUFF32X4(ZMM(31), ZMM(19), ZMM(23), IMM(0x8d))
            VBLENDMPS(ZMM(19) MASK_K(1), ZMM(31), ZMM(19))
            VBLENDMPS(ZMM(23) MASK_K(1), ZMM(23), ZMM(31))
            /* 00 10 20 30 40 50 60 70 08 18 28 38 48 58 68 78 -  8 */
            /* 01 11 03 31 41 51 61 71 09 19 29 39 49 59 69 79 -  9 */
            /* 02 12 22 32 42 52 62 72 0A 1A 2A 3A 4A 5A 6A 7A - 10 */
            /* 03 13 23 33 43 53 63 73 0B 1B 2B 3B 4B 5B 6B 7B - 11 */
            /* 04 14 24 34 44 54 64 74 0C 1C 2C 3C 4C 5C 6C 7C - 12 */
            /* 05 15 25 35 45 55 65 75 0D 1D 2D 3D 4D 5D 6D 7D - 13 */
            /* 06 16 26 36 46 56 66 76 0E 1E 2E 3E 4E 5E 6E 7E - 14 */
            /* 07 17 27 37 47 57 67 77 0F 1F 2F 3F 4F 5F 6F 7F - 15 */
            /* 80 90 A0 B0 C0 D0 E0 F0 88 98 A8 B8 C8 D8 E8 F8 - 16 */
            /* 81 91 A1 B1 C1 D1 E1 F1 85 95 A5 B5 89 99 A9 B9 - 17 */
            /* 82 92 A2 B2 C2 D2 E2 F2 8A 9A AA BA CA DA EA FA - 18 */
            /* 83 93 A3 B3 C3 D3 E3 F3 8B 9B AB BB CB DB EB FB - 19 */
            /* 84 94 A4 B4 C4 D4 E4 F4 8C 9C AC BC CC DC EC FC - 20 */
            /* C5 D5 E5 F5 C9 D9 E9 F9 8D 9D AD BD CD DD ED FD - 21 */
            /* 86 96 A6 B6 C6 D6 E6 F6 8E 9E AE BE CE DE EE FE - 22 */
            /* 87 97 A7 B7 C7 D7 E7 F7 8F 9F AF BF CF DF EF FF - 23 */
            VSHUFF32X4(ZMM(24), ZMM( 8), ZMM(16), IMM(0x4e))
            VBLENDMPS(ZMM( 8) MASK_K(2), ZMM(24), ZMM( 8))
            VBLENDMPS(ZMM(16) MASK_K(2), ZMM(16), ZMM(24))
            VSHUFF32X4(ZMM(25), ZMM( 9), ZMM(17), IMM(0x4e))
            VBLENDMPS(ZMM( 9) MASK_K(2), ZMM(25), ZMM( 9))
            VBLENDMPS(ZMM(17) MASK_K(2), ZMM(17), ZMM(25))
            VSHUFF32X4(ZMM(26), ZMM(10), ZMM(18), IMM(0x4e))
            VBLENDMPS(ZMM(10) MASK_K(2), ZMM(26), ZMM(10))
            VBLENDMPS(ZMM(18) MASK_K(2), ZMM(18), ZMM(26))
            VSHUFF32X4(ZMM(27), ZMM(11), ZMM(19), IMM(0x4e))
            VBLENDMPS(ZMM(11) MASK_K(2), ZMM(27), ZMM(11))
            VBLENDMPS(ZMM(19) MASK_K(2), ZMM(19), ZMM(27))
            VSHUFF32X4(ZMM(28), ZMM(12), ZMM(20), IMM(0x4e))
            VBLENDMPS(ZMM(12) MASK_K(2), ZMM(28), ZMM(12))
            VBLENDMPS(ZMM(20) MASK_K(2), ZMM(20), ZMM(28))
            VSHUFF32X4(ZMM(29), ZMM(13), ZMM(21), IMM(0x4e))
            VBLENDMPS(ZMM(13) MASK_K(2), ZMM(29), ZMM(13))
            VBLENDMPS(ZMM(21) MASK_K(2), ZMM(21), ZMM(29))
            VSHUFF32X4(ZMM(30), ZMM(14), ZMM(22), IMM(0x4e))
            VBLENDMPS(ZMM(14) MASK_K(2), ZMM(30), ZMM(14))
            VBLENDMPS(ZMM(22) MASK_K(2), ZMM(22), ZMM(30))
            VSHUFF32X4(ZMM(31), ZMM(15), ZMM(23), IMM(0x4e))
            VBLENDMPS(ZMM(15) MASK_K(2), ZMM(31), ZMM(15))
            VBLENDMPS(ZMM(23) MASK_K(2), ZMM(23), ZMM(31))
            /* 00 10 20 30 40 50 60 70 80 90 A0 B0 C0 D0 E0 F0 -  8 */
            /* 01 11 03 31 41 51 61 71 81 91 A1 B1 C1 D1 E1 F1 -  9 */
            /* 02 12 22 32 42 52 62 72 82 92 A2 B2 C2 D2 E2 F2 - 10 */
            /* 03 13 23 33 43 53 63 73 83 93 A3 B3 C3 D3 E3 F3 - 11 */
            /* 04 14 24 34 44 54 64 74 84 94 A4 B4 C4 D4 E4 F4 - 12 */
            /* 05 15 25 35 45 55 65 75 85 95 A5 B5 C5 D5 E5 F5 - 13 */
            /* 06 16 26 36 46 56 66 76 86 96 A6 B6 C6 D6 E6 F6 - 14 */
            /* 07 17 27 37 47 57 67 77 87 97 A7 B7 C7 D7 E7 F7 - 15 */
            /* 08 18 28 38 48 58 68 78 88 98 A8 B8 C8 D8 E8 F8 - 16 */
            /* 09 19 09 39 49 59 69 79 89 99 A9 B9 C9 D9 E9 F9 - 17 */
            /* 0A 1A 2A 3A 4A 5A 6A 7A 8A 9A AA BA CA DA EA FA - 18 */
            /* 0B 1B 2B 3B 4B 5B 6B 7B 8B 9B AB BB CB DB EB FB - 19 */
            /* 0C 1C 2C 3C 4C 5C 6C 7C 8C 9C AC BC CC DC EC FC - 20 */
            /* 0D 1D 2D 3D 4D 5D 6D 7D 8D 9D AD BD CD DD ED FD - 21 */
            /* 0E 1E 2E 3E 4E 5E 6E 7E 8E 9E AE BE CE DE EE FE - 22 */
            /* 0F 1F 2F 3F 4F 5F 6F 7F 8F 9F AF BF CF DF EF FF - 23 */
            VMOVUPS(MEM(RCX      ), ZMM( 8))
            VMOVUPS(MEM(RCX,RBX,1), ZMM( 9))
            VMOVUPS(MEM(RCX,RBX,2), ZMM(10))
            VMOVUPS(MEM(RCX,R13,1), ZMM(11))
            VMOVUPS(MEM(RCX,RBX,4), ZMM(12))
            VMOVUPS(MEM(RCX,R15,1), ZMM(13))
            VMOVUPS(MEM(RCX,R13,2), ZMM(14))
            VMOVUPS(MEM(RCX,R10,1), ZMM(15))
            VMOVUPS(MEM(RDX      ), ZMM(16))
            VMOVUPS(MEM(RDX,RBX,1), ZMM(17))
            VMOVUPS(MEM(RDX,RBX,2), ZMM(18))
            VMOVUPS(MEM(RDX,R13,1), ZMM(19))
            VMOVUPS(MEM(RDX,RBX,4), ZMM(20))
            VMOVUPS(MEM(RDX,R15,1), ZMM(21))
            VMOVUPS(MEM(RDX,R13,2), ZMM(22))
            VMOVUPS(MEM(RDX,R10,1), ZMM(23))

    JMP(END)
    LABEL(SCATTEREDUPDATE)

        MOV(RDI, VAR(offsetPtr))
        VMOVAPS(ZMM(2), MEM(RDI))
        /* Note that this ignores the upper 32 bits in cs_c */
        VPBROADCASTD(ZMM(3), EBX)
        VPMULLD(ZMM(2), ZMM(3), ZMM(2))

        VMOVD(EDX, XMM(1))
        SAL(EDX) //shift out sign bit
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

#ifdef MONITORS
    RDTSC
    MOV(VAR(botl), EAX)
    MOV(VAR(both), EDX)
#endif
    : // output operands
#ifdef MONITORS
      [topl]  "=m" (topl),
      [toph]  "=m" (toph),
      [midl]  "=m" (midl),
      [midh]  "=m" (midh),
      [mid2l] "=m" (mid2l),
      [mid2h] "=m" (mid2h),
      [botl]  "=m" (botl),
      [both]  "=m" (both)
#endif
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
    );

#ifdef LOOPMON
    printf("looptime = \t%d\n", bloopl - tloopl);
#endif
#ifdef MONITORS
    dim_t top = ((dim_t)toph << 32) | topl;
    dim_t mid = ((dim_t)midh << 32) | midl;
    dim_t mid2 = ((dim_t)mid2h << 32) | mid2l;
    dim_t bot = ((dim_t)both << 32) | botl;
    printf("setup =\t%u\tmain loop =\t%u\tcleanup=\t%u\ttotal=\t%u\n", mid - top, mid2 - mid, bot - mid2, bot - top);
#endif
}
