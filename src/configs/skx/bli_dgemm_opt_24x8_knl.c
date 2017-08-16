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

#define PREFETCH_C_L2 1

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
    VMULPD(ZMM(NUM), ZMM(NUM), ZMM(0)) \
    VGATHERQPD(ZMM(3), MEM(RCX,ZMM(2),8)) \
    VFMADD231PD(ZMM(NUM), ZMM(3), ZMM(1)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),8), ZMM(NUM)) \
    ADD(RCX, RAX)

#define UPDATE_C_BZ_ROW_SCATTERED(NUM) \
\
    VMULPD(ZMM(NUM), ZMM(NUM), ZMM(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),8), ZMM(NUM)) \
    ADD(RCX, RAX)

#define PREFETCH_A_L1_1(n) PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*24*8))
#define PREFETCH_A_L1_2(n) PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*24*8+64))
#define PREFETCH_A_L1_3(n) PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*24*8+128))

#define PREFETCH_B_L1(n) PREFETCH(0, MEM(RBX,(B_L1_PREFETCH_DIST+n)*8*8))

#define PREFETCH_C(n) \
\
    PREFETCHW##n(MEM(RCX      )) \
    PREFETCHW##n(MEM(RCX,R12,1)) \
    PREFETCHW##n(MEM(RCX,R12,2)) \
    PREFETCHW##n(MEM(RCX,R13,1)) \
    PREFETCHW##n(MEM(RCX,R12,4)) \
    PREFETCHW##n(MEM(RCX,R14,1)) \
    PREFETCHW##n(MEM(RCX,R13,2)) \
    PREFETCHW##n(MEM(RCX,R15,1)) \
    LEA(RDX, MEM(RCX,R12,8)) \
    PREFETCHW##n(MEM(RDX      )) \
    PREFETCHW##n(MEM(RDX,R12,1)) \
    PREFETCHW##n(MEM(RDX,R12,2)) \
    PREFETCHW##n(MEM(RDX,R13,1)) \
    PREFETCHW##n(MEM(RDX,R12,4)) \
    PREFETCHW##n(MEM(RDX,R14,1)) \
    PREFETCHW##n(MEM(RDX,R13,2)) \
    PREFETCHW##n(MEM(RDX,R15,1)) \
    LEA(RDI, MEM(RDX,R12,8)) \
    PREFETCHW##n(MEM(RDI      )) \
    PREFETCHW##n(MEM(RDI,R12,1)) \
    PREFETCHW##n(MEM(RDI,R12,2)) \
    PREFETCHW##n(MEM(RDI,R13,1)) \
    PREFETCHW##n(MEM(RDI,R12,4)) \
    PREFETCHW##n(MEM(RDI,R14,1)) \
    PREFETCHW##n(MEM(RDI,R13,2)) \
    PREFETCHW##n(MEM(RDI,R15,1))

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
    VFMADD231PD(ZMM(11), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 3)*8)) \
    VFMADD231PD(ZMM(12), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 4)*8)) \
    PREFETCH_A_L1_1(n) \
    VFMADD231PD(ZMM(13), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 5)*8)) \
    VFMADD231PD(ZMM(14), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 6)*8)) \
    VFMADD231PD(ZMM(15), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 7)*8)) \
    VFMADD231PD(ZMM(16), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 8)*8)) \
    VFMADD231PD(ZMM(17), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 9)*8)) \
    PREFETCH_A_L1_2(n) \
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
    PREFETCH_B_L1(n) \
    VFMADD231PD(ZMM(28), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+20)*8)) \
    VFMADD231PD(ZMM(29), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+21)*8)) \
    VFMADD231PD(ZMM(30), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+22)*8)) \
    VFMADD231PD(ZMM(31), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+23)*8))

//This is an array used for the scatter/gather instructions.
static int64_t offsets[8] __attribute__((aligned(64))) = {0,1,2,3,4,5,6,7};

void bli_dgemm_opt_24x8_knl(
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

    __asm__ volatile
    (

    VPXORD(ZMM(8), ZMM(8), ZMM(8)) //clear out registers
    VMOVAPD(ZMM( 7), ZMM(8))
    VMOVAPD(ZMM( 9), ZMM(8))
    VMOVAPD(ZMM(10), ZMM(8))   MOV(RSI, VAR(k)) //loop index
    VMOVAPD(ZMM(11), ZMM(8))   MOV(RAX, VAR(a)) //load address of a
    VMOVAPD(ZMM(12), ZMM(8))   MOV(RBX, VAR(b)) //load address of b
    VMOVAPD(ZMM(13), ZMM(8))   MOV(RCX, VAR(c)) //load address of c
    VMOVAPD(ZMM(14), ZMM(8))   VMOVAPD(ZMM(0), MEM(RBX)) //pre-load b
    VMOVAPD(ZMM(15), ZMM(8))
    VMOVAPD(ZMM(16), ZMM(8))
    VMOVAPD(ZMM(17), ZMM(8))   MOV(R12, VAR(rs_c))      //rs_c
    VMOVAPD(ZMM(18), ZMM(8))   LEA(R13, MEM(R12,R12,2)) //*3
    VMOVAPD(ZMM(19), ZMM(8))   LEA(R14, MEM(R12,R12,4)) //*5
    VMOVAPD(ZMM(20), ZMM(8))   LEA(R15, MEM(R13,R12,4)) //*7
    VMOVAPD(ZMM(21), ZMM(8))
    VMOVAPD(ZMM(22), ZMM(8))
    VMOVAPD(ZMM(23), ZMM(8))
    VMOVAPD(ZMM(24), ZMM(8))
    VMOVAPD(ZMM(25), ZMM(8))   MOV(R8, IMM(4*24*8))     //offset for 4 iterations
    VMOVAPD(ZMM(26), ZMM(8))   LEA(R9, MEM(R8,R8,2))    //*3
    VMOVAPD(ZMM(27), ZMM(8))   LEA(R10, MEM(R8,R8,4))   //*5
    VMOVAPD(ZMM(28), ZMM(8))   LEA(R11, MEM(R9,R8,4))   //*7
    VMOVAPD(ZMM(29), ZMM(8))
    VMOVAPD(ZMM(30), ZMM(8))
    VMOVAPD(ZMM(31), ZMM(8))

    TEST(RSI, RSI)
    JZ(POSTACCUM)

    CMP(RSI, IMM(64))
    JL(K_SMALL)

#if PREFETCH_C_L2
    PREFETCH_C(1)
#endif

    MOV(RDI, RSI)
    AND(RSI, IMM(31))
    SAR(RDI, IMM(5))
    SUB(RDI, IMM(1))

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

        SUB(RDI, IMM(1))

    JNZ(MAIN_LOOP)

    PREFETCH_C(0)

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

    JMP(TAIL_LOOP)
    LABEL(K_SMALL)

    PREFETCH_C(0)

    LOOP_ALIGN
    LABEL(TAIL_LOOP)

        SUBITER(0,1,0,RAX)
        VMOVAPD(ZMM(0), ZMM(1))
        ADD(RAX, IMM(24*8))
        ADD(RBX, IMM( 8*8))

        SUB(RSI, IMM(1))

    JNZ(TAIL_LOOP)

    LABEL(POSTACCUM)

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSD(ZMM(0), MEM(RAX))
    VBROADCASTSD(ZMM(1), MEM(RBX))

    // Check if C is row stride. If not, jump to the slow scattered update
    MOV(RAX, VAR(rs_c))
    LEA(RAX, MEM(,RAX,8))
    MOV(RBX, VAR(cs_c))
    LEA(RDI, MEM(RAX,RAX,2))
    CMP(RBX, IMM(1))
    JNE(SCATTEREDUPDATE)

    VCOMISD(XMM(1), XMM(7))
    JE(COLSTORBZ)

    UPDATE_C_FOUR_ROWS( 8, 9,10,11)
    UPDATE_C_FOUR_ROWS(12,13,14,15)
    UPDATE_C_FOUR_ROWS(16,17,18,19)
    UPDATE_C_FOUR_ROWS(20,21,22,23)
    UPDATE_C_FOUR_ROWS(24,25,26,27)
    UPDATE_C_FOUR_ROWS(28,29,30,31)

    JMP(END)

    LABEL(COLSTORBZ)

    UPDATE_C_BZ_FOUR_ROWS( 8, 9,10,11)
    UPDATE_C_BZ_FOUR_ROWS(12,13,14,15)
    UPDATE_C_BZ_FOUR_ROWS(16,17,18,19)
    UPDATE_C_BZ_FOUR_ROWS(20,21,22,23)
    UPDATE_C_BZ_FOUR_ROWS(24,25,26,27)
    UPDATE_C_BZ_FOUR_ROWS(28,29,30,31)

    JMP(END)

    LABEL(SCATTEREDUPDATE)

    MOV(RDI, VAR(offsetPtr))
    VMOVDQA64(ZMM(2), MEM(RDI))
    VPBROADCASTQ(ZMM(3), RBX)
    VPMULLQ(ZMM(2), ZMM(3), ZMM(2))

    VCOMISD(XMM(1), XMM(7))
    JE(SCATTERBZ)

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
