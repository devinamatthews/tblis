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

#include "../knl/bli_avx512_macros.h"

#define A_L1_PREFETCH_DIST 18
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

#define PREFETCH_A_L1(n,off) PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*24*4+off))
#define PREFETCH_B_L1(n,off) PREFETCH(0, MEM(RBX,(B_L1_PREFETCH_DIST+n)*16*4+off))

//
// n: index in unrolled loop
//
// a: ZMM register to load into
// b: ZMM register to read from
//
// ...: addressing for A, except for offset
//
#define SUBITER4(n,...) \
\
    VMOVAPD(ZMM(0), MEM(RBX,(n+0)*64)) \
    VMOVAPD(ZMM(1), MEM(RBX,(n+1)*64)) \
    VMOVAPD(ZMM(2), MEM(RBX,(n+2)*64)) \
    VMOVAPD(ZMM(3), MEM(RBX,(n+3)*64)) \
    V4FMADDPS(ZMM( 8), ZMM(0), MEM(__VA_ARGS__, 0*4*4)) \
    V4FMADDPS(ZMM( 9), ZMM(0), MEM(__VA_ARGS__, 1*4*4)) \
    V4FMADDPS(ZMM(10), ZMM(0), MEM(__VA_ARGS__, 2*4*4)) \
    PREFETCH_A_L1(n,0*64) \
    V4FMADDPS(ZMM(11), ZMM(0), MEM(__VA_ARGS__, 3*4*4)) \
    V4FMADDPS(ZMM(12), ZMM(0), MEM(__VA_ARGS__, 4*4*4)) \
    PREFETCH_A_L1(n,1*64) \
    V4FMADDPS(ZMM(13), ZMM(0), MEM(__VA_ARGS__, 5*4*4)) \
    V4FMADDPS(ZMM(14), ZMM(0), MEM(__VA_ARGS__, 6*4*4)) \
    PREFETCH_B_L1(n,0*64) \
    V4FMADDPS(ZMM(15), ZMM(0), MEM(__VA_ARGS__, 7*4*4)) \
    V4FMADDPS(ZMM(16), ZMM(0), MEM(__VA_ARGS__, 8*4*4)) \
    PREFETCH_A_L1(n,2*64) \
    V4FMADDPS(ZMM(17), ZMM(0), MEM(__VA_ARGS__, 9*4*4)) \
    V4FMADDPS(ZMM(18), ZMM(0), MEM(__VA_ARGS__,10*4*4)) \
    PREFETCH_B_L1(n,1*64) \
    V4FMADDPS(ZMM(19), ZMM(0), MEM(__VA_ARGS__,11*4*4)) \
    V4FMADDPS(ZMM(20), ZMM(0), MEM(__VA_ARGS__,12*4*4)) \
    PREFETCH_A_L1(n,3*64) \
    V4FMADDPS(ZMM(21), ZMM(0), MEM(__VA_ARGS__,13*4*4)) \
    V4FMADDPS(ZMM(22), ZMM(0), MEM(__VA_ARGS__,14*4*4)) \
    PREFETCH_B_L1(n,2*64) \
    V4FMADDPS(ZMM(23), ZMM(0), MEM(__VA_ARGS__,15*4*4)) \
    V4FMADDPS(ZMM(24), ZMM(0), MEM(__VA_ARGS__,16*4*4)) \
    PREFETCH_A_L1(n,4*64) \
    V4FMADDPS(ZMM(25), ZMM(0), MEM(__VA_ARGS__,17*4*4)) \
    V4FMADDPS(ZMM(26), ZMM(0), MEM(__VA_ARGS__,18*4*4)) \
    PREFETCH_B_L1(n,3*64) \
    V4FMADDPS(ZMM(27), ZMM(0), MEM(__VA_ARGS__,19*4*4)) \
    V4FMADDPS(ZMM(28), ZMM(0), MEM(__VA_ARGS__,20*4*4)) \
    PREFETCH_A_L1(n,5*64) \
    V4FMADDPS(ZMM(29), ZMM(0), MEM(__VA_ARGS__,21*4*4)) \
    V4FMADDPS(ZMM(30), ZMM(0), MEM(__VA_ARGS__,22*4*4)) \
    V4FMADDPS(ZMM(31), ZMM(0), MEM(__VA_ARGS__,23*4*4))

#define SUBITER(...) \
\
    VMOVAPD(ZMM(0), MEM(RBX)) \
    VFMADD231PS(ZMM( 8), ZMM(0), MEM_1TO16(__VA_ARGS__, 0*4)) \
    VFMADD231PS(ZMM( 9), ZMM(0), MEM_1TO16(__VA_ARGS__, 1*4)) \
    VFMADD231PS(ZMM(10), ZMM(0), MEM_1TO16(__VA_ARGS__, 2*4)) \
    VFMADD231PS(ZMM(11), ZMM(0), MEM_1TO16(__VA_ARGS__, 3*4)) \
    VFMADD231PS(ZMM(12), ZMM(0), MEM_1TO16(__VA_ARGS__, 4*4)) \
    VFMADD231PS(ZMM(13), ZMM(0), MEM_1TO16(__VA_ARGS__, 5*4)) \
    PREFETCH_A_L1(n,0) \
    VFMADD231PS(ZMM(14), ZMM(0), MEM_1TO16(__VA_ARGS__, 6*4)) \
    VFMADD231PS(ZMM(15), ZMM(0), MEM_1TO16(__VA_ARGS__, 7*4)) \
    VFMADD231PS(ZMM(16), ZMM(0), MEM_1TO16(__VA_ARGS__, 8*4)) \
    VFMADD231PS(ZMM(17), ZMM(0), MEM_1TO16(__VA_ARGS__, 9*4)) \
    VFMADD231PS(ZMM(18), ZMM(0), MEM_1TO16(__VA_ARGS__,10*4)) \
    VFMADD231PS(ZMM(19), ZMM(0), MEM_1TO16(__VA_ARGS__,11*4)) \
    PREFETCH_A_L1(n,64) \
    VFMADD231PS(ZMM(20), ZMM(0), MEM_1TO16(__VA_ARGS__,12*4)) \
    VFMADD231PS(ZMM(21), ZMM(0), MEM_1TO16(__VA_ARGS__,13*4)) \
    VFMADD231PS(ZMM(22), ZMM(0), MEM_1TO16(__VA_ARGS__,14*4)) \
    VFMADD231PS(ZMM(23), ZMM(0), MEM_1TO16(__VA_ARGS__,15*4)) \
    VFMADD231PS(ZMM(24), ZMM(0), MEM_1TO16(__VA_ARGS__,16*4)) \
    VFMADD231PS(ZMM(25), ZMM(0), MEM_1TO16(__VA_ARGS__,17*4)) \
    PREFETCH_B_L1(n,0) \
    VFMADD231PS(ZMM(26), ZMM(0), MEM_1TO16(__VA_ARGS__,18*4)) \
    VFMADD231PS(ZMM(27), ZMM(0), MEM_1TO16(__VA_ARGS__,19*4)) \
    VFMADD231PS(ZMM(28), ZMM(0), MEM_1TO16(__VA_ARGS__,20*4)) \
    VFMADD231PS(ZMM(29), ZMM(0), MEM_1TO16(__VA_ARGS__,21*4)) \
    VFMADD231PS(ZMM(30), ZMM(0), MEM_1TO16(__VA_ARGS__,22*4)) \
    VFMADD231PS(ZMM(31), ZMM(0), MEM_1TO16(__VA_ARGS__,23*4))

#define PREFETCH_C_16(level,vsib1,off) \
        KXNORW(K(1), K(0), K(0)) \
        KXNORW(K(2), K(0), K(0)) \
        VSCATTERPFDPS(level, MEM(RCX,ZMM(vsib1),4,off+0*64) MASK_K(1)) \
        VSCATTERPFDPS(level, MEM(RCX,ZMM(vsib1),4,off+1*64) MASK_K(2))

#define PREFETCH_C_24(level,vsib1,vsib2,off) \
        KXNORW(K(1), K(0), K(0)) \
        KXNORW(K(2), K(0), K(0)) \
        VSCATTERPFDPS(level, MEM(RCX,ZMM(vsib1),4,off) MASK_K(1)) \
        VSCATTERPFDPD(level, MEM(RCX,YMM(vsib2),4,off) MASK_K(2))

//This is an array used for the scatter/gather instructions.
static int32_t offsets[32] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};

void bli_sgemm_opt_24x16(
                          dim_t            k_,
                          float* restrict  alpha,
                          float* restrict  a,
                          float* restrict  b,
                          float* restrict  beta,
                          float* restrict  c, inc_t rs_c_, inc_t cs_c_,
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

    __asm__ __volatile__ (

    VPXORD(ZMM(8), ZMM(8), ZMM(8)) //clear out registers
    VMOVAPS(ZMM( 9), ZMM(8))
    VMOVAPS(ZMM(10), ZMM(8))
    VMOVAPS(ZMM(11), ZMM(8))
    VMOVAPS(ZMM(12), ZMM(8))
    VMOVAPS(ZMM(13), ZMM(8))
    VMOVAPS(ZMM(14), ZMM(8))
    VMOVAPS(ZMM(15), ZMM(8))
    VMOVAPS(ZMM(16), ZMM(8))
    VMOVAPS(ZMM(17), ZMM(8))
    VMOVAPS(ZMM(18), ZMM(8))
    VMOVAPS(ZMM(19), ZMM(8))
    VMOVAPS(ZMM(20), ZMM(8))
    VMOVAPS(ZMM(21), ZMM(8))
    VMOVAPS(ZMM(22), ZMM(8))
    VMOVAPS(ZMM(23), ZMM(8))
    VMOVAPS(ZMM(24), ZMM(8))
    VMOVAPS(ZMM(25), ZMM(8))
    VMOVAPS(ZMM(26), ZMM(8))
    VMOVAPS(ZMM(27), ZMM(8))
    VMOVAPS(ZMM(28), ZMM(8))
    VMOVAPS(ZMM(29), ZMM(8))
    VMOVAPS(ZMM(30), ZMM(8))
    VMOVAPS(ZMM(31), ZMM(8))

    MOV(R12, VAR(rs_c))
    LEA(R12, MEM(,R12,4))
    MOV(RSI, VAR(k)) //loop index
    MOV(RAX, VAR(a)) //load address of a
    MOV(RBX, VAR(b)) //load address of b
    MOV(RCX, VAR(c)) //load address of c

    MOV(R15, VAR(offsetPtr))
    VMOVDQA32(ZMM(4), MEM(R15))
    VMOVDQA32(ZMM(5), MEM(R15,64))

    MOV(R8, IMM(24*4*4))     //offset for 4 iterations
    LEA(R9, MEM(R8,R8,2))    //*3
    LEA(R10, MEM(R8,R8,4))   //*5
    LEA(R11, MEM(R9,R8,4))   //*7

    //prefetch C into L2

    CMP(R12, IMM(4))
    JE(COLSTORPF2)

        VPMULLD(ZMM(4), ZMM(4), VAR_1TO16(rs_c))
        VPMULLD(ZMM(5), ZMM(5), VAR_1TO16(rs_c))
        PREFETCH_C_24(1,4,5,64)

    JMP(PFDONE2)
    LABEL(COLSTORPF2)

        VPMULLD(ZMM(4), ZMM(4), VAR_1TO16(cs_c))
        VPMULLD(ZMM(5), ZMM(5), VAR_1TO16(cs_c))
        PREFETCH_C_16(1,5,0)

    LABEL(PFDONE2)

    MOV(RDI, RSI)
    AND(RDI, IMM(63))
    SAR(RSI, IMM(6))
    JZ(PREFETCHC)

    LOOP_ALIGN
    LABEL(MAIN_LOOP)

        SUBITER4( 0,RAX      )
        SUBITER4( 4,RAX,R8, 1)
        SUBITER4( 8,RAX,R8, 2)
        SUBITER4(12,RAX,R9, 1)
        SUBITER4(16,RAX,R8, 4)
        SUBITER4(20,RAX,R10,1)
        SUBITER4(24,RAX,R9, 2)
        SUBITER4(28,RAX,R11,1)

        ADD(RAX, IMM(32*24*4))
        ADD(RBX, IMM(32*16*4))

        SUB(RSI, IMM(1))

    JNZ(MAIN_LOOP)

    //prefetch C into L1

    LABEL(PREFETCHC)

    CMP(R12, IMM(4))
    JE(COLSTORPF)

        PREFETCH_C_24(0,4,5,0)

    JMP(PFDONE)
    LABEL(COLSTORPF)

        PREFETCH_C_16(0,4,0)

    LABEL(PFDONE)

    MOV(RSI, RDI)
    AND(RSI, IMM(3))
    SAR(RDI, IMM(2))
    JZ(CLEANUP)

    LOOP_ALIGN
    LABEL(TAIL_LOOP)

        SUBITER4(0,RAX)

        ADD(RAX, IMM(4*24*4))
        ADD(RBX, IMM(4*16*4))

        SUB(RDI, IMM(1))

    JNZ(TAIL_LOOP)

    LABEL(CLEANUP)

    TEST(RSI,RSI)
    JZ(POSTACCUM)

    LOOP_ALIGN
    LABEL(CLEANUP_LOOP)

        SUBITER(RAX)

        ADD(RAX, IMM(24*4))
        ADD(RBX, IMM(16*4))

        SUB(RSI, IMM(1))

    JNZ(CLEANUP_LOOP)

    LABEL(POSTACCUM)

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSD(ZMM(0), MEM(RAX))
    VBROADCASTSD(ZMM(1), MEM(RBX))

    MOV(RAX, VAR(rs_c))
    LEA(RAX, MEM(,RAX,4))
    MOV(RBX, VAR(cs_c))
    LEA(RBX, MEM(,RBX,4))
    LEA(RDI, MEM(RAX,RAX,2))

    // Check if C is row stride.
    CMP(RBX, IMM(4))
    JNE(SCATTEREDUPDATE)

        VMOVQ(RDX, XMM(1))
        SAL1(RDX) //shift out sign bit
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
    LABEL(SCATTEREDUPDATE)

        VPBROADCASTD(ZMM(5), VAR(rs_c))
        VPMULLD(ZMM(2), ZMM(5), MEM(R15))

        VMOVQ(RDX, XMM(1))
        SAL1(RDX) //shift out sign bit
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
    );
}

void bli_sgemm_opt_16x24(
                          dim_t            k_,
                          float* restrict  alpha,
                          float* restrict  b,
                          float* restrict  a,
                          float* restrict  beta,
                          float* restrict  c, inc_t cs_c_, inc_t rs_c_,
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

    __asm__ __volatile__ (

    VPXORD(ZMM(8), ZMM(8), ZMM(8)) //clear out registers
    VMOVAPS(ZMM( 9), ZMM(8))
    VMOVAPS(ZMM(10), ZMM(8))
    VMOVAPS(ZMM(11), ZMM(8))
    VMOVAPS(ZMM(12), ZMM(8))
    VMOVAPS(ZMM(13), ZMM(8))
    VMOVAPS(ZMM(14), ZMM(8))
    VMOVAPS(ZMM(15), ZMM(8))
    VMOVAPS(ZMM(16), ZMM(8))
    VMOVAPS(ZMM(17), ZMM(8))
    VMOVAPS(ZMM(18), ZMM(8))
    VMOVAPS(ZMM(19), ZMM(8))
    VMOVAPS(ZMM(20), ZMM(8))
    VMOVAPS(ZMM(21), ZMM(8))
    VMOVAPS(ZMM(22), ZMM(8))
    VMOVAPS(ZMM(23), ZMM(8))
    VMOVAPS(ZMM(24), ZMM(8))
    VMOVAPS(ZMM(25), ZMM(8))
    VMOVAPS(ZMM(26), ZMM(8))
    VMOVAPS(ZMM(27), ZMM(8))
    VMOVAPS(ZMM(28), ZMM(8))
    VMOVAPS(ZMM(29), ZMM(8))
    VMOVAPS(ZMM(30), ZMM(8))
    VMOVAPS(ZMM(31), ZMM(8))

    MOV(R12, VAR(rs_c))
    LEA(R12, MEM(,R12,4))
    MOV(RSI, VAR(k)) //loop index
    MOV(RAX, VAR(a)) //load address of a
    MOV(RBX, VAR(b)) //load address of b
    MOV(RCX, VAR(c)) //load address of c

    MOV(R15, VAR(offsetPtr))
    VMOVDQA32(ZMM(4), MEM(R15))
    VMOVDQA32(ZMM(5), MEM(R15,64))

    MOV(R8, IMM(24*4*4))     //offset for 4 iterations
    LEA(R9, MEM(R8,R8,2))    //*3
    LEA(R10, MEM(R8,R8,4))   //*5
    LEA(R11, MEM(R9,R8,4))   //*7

    //prefetch C into L2

    CMP(R12, IMM(4))
    JE(.ROWSTORPF2)

        VPMULLD(ZMM(4), ZMM(4), VAR_1TO16(rs_c))
        VPMULLD(ZMM(5), ZMM(5), VAR_1TO16(rs_c))
        MOV(RDX, RCX)
        LEA(RCX, MEM(RCX,R12,8))
        LEA(RCX, MEM(RCX,R12,8))
        LEA(RCX, MEM(RCX,R12,8))
        PREFETCH_C_24(1,4,5,0)
        MOV(RCX, RDX)

    JMP(.PFDONE2)
    LABEL(.ROWSTORPF2)

        VPMULLD(ZMM(4), ZMM(4), VAR_1TO16(cs_c))
        PREFETCH_C_16(1,4,96)

    LABEL(.PFDONE2)

    MOV(RDI, RSI)
    AND(RDI, IMM(63))
    SAR(RSI, IMM(6))
    JZ(.PREFETCHC)

    LOOP_ALIGN
    LABEL(.MAIN_LOOP)

        SUBITER4( 0,RAX      )
        SUBITER4( 4,RAX,R8, 1)
        SUBITER4( 8,RAX,R8, 2)
        SUBITER4(12,RAX,R9, 1)
        SUBITER4(16,RAX,R8, 4)
        SUBITER4(20,RAX,R10,1)
        SUBITER4(24,RAX,R9, 2)
        SUBITER4(28,RAX,R11,1)

        ADD(RAX, IMM(32*24*4))
        ADD(RBX, IMM(32*16*4))

        SUB(RSI, IMM(1))

    JNZ(.MAIN_LOOP)

    //prefetch C into L1

    LABEL(.PREFETCHC)

    CMP(R12, IMM(4))
    JE(.ROWSTORPF)

        PREFETCH_C_24(0,4,5,0)

    JMP(.PFDONE)
    LABEL(.ROWSTORPF)

        PREFETCH_C_16(0,4,0)

    LABEL(.PFDONE)

    MOV(RSI, RDI)
    AND(RSI, IMM(3))
    SAR(RDI, IMM(2))
    JZ(.CLEANUP)

    LOOP_ALIGN
    LABEL(.TAIL_LOOP)

        SUBITER4(0,RAX)

        ADD(RAX, IMM(4*24*4))
        ADD(RBX, IMM(4*16*4))

        SUB(RDI, IMM(1))

    JNZ(.TAIL_LOOP)

    LABEL(.CLEANUP)

    TEST(RSI,RSI)
    JZ(.POSTACCUM)

    LOOP_ALIGN
    LABEL(.CLEANUP_LOOP)

        SUBITER(RAX)

        ADD(RAX, IMM(24*4))
        ADD(RBX, IMM(16*4))

        SUB(RSI, IMM(1))

    JNZ(.CLEANUP_LOOP)

    LABEL(.POSTACCUM)

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSD(ZMM(0), MEM(RAX))
    VBROADCASTSD(ZMM(1), MEM(RBX))

    MOV(RAX, VAR(rs_c))
    LEA(RAX, MEM(,RAX,4))
    MOV(RBX, VAR(cs_c))
    LEA(RBX, MEM(,RBX,4))
    LEA(RDI, MEM(RAX,RAX,2))

    // Check if C is row stride.
    CMP(RBX, IMM(4))
    JNE(.SCATTEREDUPDATE)

        VMOVQ(RDX, XMM(1))
        SAL1(RDX) //shift out sign bit
        JZ(.COLSTORBZ)

            UPDATE_C_FOUR_ROWS( 8, 9,10,11)
            UPDATE_C_FOUR_ROWS(12,13,14,15)
            UPDATE_C_FOUR_ROWS(16,17,18,19)
            UPDATE_C_FOUR_ROWS(20,21,22,23)
            UPDATE_C_FOUR_ROWS(24,25,26,27)
            UPDATE_C_FOUR_ROWS(28,29,30,31)

        JMP(.END)
        LABEL(.COLSTORBZ)

            UPDATE_C_BZ_FOUR_ROWS( 8, 9,10,11)
            UPDATE_C_BZ_FOUR_ROWS(12,13,14,15)
            UPDATE_C_BZ_FOUR_ROWS(16,17,18,19)
            UPDATE_C_BZ_FOUR_ROWS(20,21,22,23)
            UPDATE_C_BZ_FOUR_ROWS(24,25,26,27)
            UPDATE_C_BZ_FOUR_ROWS(28,29,30,31)

    JMP(.END)
    LABEL(.SCATTEREDUPDATE)

        VPBROADCASTD(ZMM(5), VAR(rs_c))
        VPMULLD(ZMM(2), ZMM(5), MEM(R15))

        VMOVQ(RDX, XMM(1))
        SAL1(RDX) //shift out sign bit
        JZ(.SCATTERBZ)

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

        JMP(.END)
        LABEL(.SCATTERBZ)

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

    LABEL(.END)

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
    );
}
