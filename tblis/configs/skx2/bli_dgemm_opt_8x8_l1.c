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

#define A_L1_PREFETCH_DIST 12

#define LOOP_ALIGN ALIGN16

#define UPDATE_C(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VFMADD231PD(ZMM(R1), ZMM(1), MEM(RCX)) \
    VFMADD231PD(ZMM(R2), ZMM(1), MEM(RCX,RAX,1)) \
    VFMADD231PD(ZMM(R3), ZMM(1), MEM(RCX,RAX,2)) \
    VFMADD231PD(ZMM(R4), ZMM(1), MEM(RCX,RDX,1)) \
    VMOVUPD(MEM(RCX), ZMM(R1)) \
    VMOVUPD(MEM(RCX,RAX,1), ZMM(R2)) \
    VMOVUPD(MEM(RCX,RAX,2), ZMM(R3)) \
    VMOVUPD(MEM(RCX,RDX,1), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,4))

#define UPDATE_C_BZ(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VMOVUPD(MEM(RCX), ZMM(R1)) \
    VMOVUPD(MEM(RCX,RAX,1), ZMM(R2)) \
    VMOVUPD(MEM(RCX,RAX,2), ZMM(R3)) \
    VMOVUPD(MEM(RCX,RDX,1), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,4))

#define UPDATE_C_ROW_SCATTERED(R1,R2,R3,R4) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VGATHERQPD(ZMM(6) MASK_K(1), MEM(RCX,ZMM(2),8)) \
    VFMADD231PD(ZMM(R1), ZMM(6), ZMM(1)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),8) MASK_K(2), ZMM(R1)) \
\
    LEA(RCX, MEM(RCX,RAX,1)) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VGATHERQPD(ZMM(6) MASK_K(1), MEM(RCX,ZMM(2),8)) \
    VFMADD231PD(ZMM(R2), ZMM(6), ZMM(1)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),8) MASK_K(2), ZMM(R2)) \
\
    LEA(RCX, MEM(RCX,RAX,1)) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VGATHERQPD(ZMM(6) MASK_K(1), MEM(RCX,ZMM(2),8)) \
    VFMADD231PD(ZMM(R3), ZMM(6), ZMM(1)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),8) MASK_K(2), ZMM(R3)) \
\
    LEA(RCX, MEM(RCX,RAX,1)) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VGATHERQPD(ZMM(6) MASK_K(1), MEM(RCX,ZMM(2),8)) \
    VFMADD231PD(ZMM(R4), ZMM(6), ZMM(1)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),8) MASK_K(2), ZMM(R4)) \
\
    LEA(RCX, MEM(RCX,RAX,1))

#define UPDATE_C_BZ_ROW_SCATTERED(R1,R2,R3,R4) \
\
    KXNORW(K(1), K(0), K(0)) \
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),8) MASK_K(1), ZMM(R1)) \
\
    LEA(RCX, MEM(RCX,RAX,1)) \
\
    KXNORW(K(1), K(0), K(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),8) MASK_K(1), ZMM(R2)) \
\
    LEA(RCX, MEM(RCX,RAX,1)) \
\
    KXNORW(K(1), K(0), K(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),8) MASK_K(1), ZMM(R3)) \
\
    LEA(RCX, MEM(RCX,RAX,1)) \
\
    KXNORW(K(1), K(0), K(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),8) MASK_K(1), ZMM(R4)) \
\
    LEA(RCX, MEM(RCX,RAX,1))

#define PREFETCH_C_L1 \
\
    PREFETCHW0(MEM(RCX,     )) \
    PREFETCHW0(MEM(RCX,R12,1)) \
    PREFETCHW0(MEM(RCX,R12,2)) \
    PREFETCHW0(MEM(RCX,R13,1)) \
    PREFETCHW0(MEM(RCX,R12,4)) \
    PREFETCHW0(MEM(RCX,R14,1)) \
    PREFETCHW0(MEM(RCX,R13,2)) \
    PREFETCHW0(MEM(RCX,R15,1))

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
    PREFETCH(0, MEM(RAX,A_L1_PREFETCH_DIST*8*8+n*64)) \
    \
    VFMADD231PD(ZMM(24), ZMM(0), MEM_1TO8(RAX,(8*n+0)*8)) \
    VFMADD231PD(ZMM(25), ZMM(0), MEM_1TO8(RAX,(8*n+1)*8)) \
    VFMADD231PD(ZMM(26), ZMM(0), MEM_1TO8(RAX,(8*n+2)*8)) \
    VFMADD231PD(ZMM(27), ZMM(0), MEM_1TO8(RAX,(8*n+3)*8)) \
    VFMADD231PD(ZMM(28), ZMM(0), MEM_1TO8(RAX,(8*n+4)*8)) \
    VFMADD231PD(ZMM(29), ZMM(0), MEM_1TO8(RAX,(8*n+5)*8)) \
    VFMADD231PD(ZMM(30), ZMM(0), MEM_1TO8(RAX,(8*n+6)*8)) \
    VFMADD231PD(ZMM(31), ZMM(0), MEM_1TO8(RAX,(8*n+7)*8)) \
    \
    VMOVAPD(ZMM(0), MEM(RBX,(8*n+0)*8))

//This is an array used for the scatter/gather instructions.
static int64_t offsets[16] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15};

void bli_dgemm_opt_8x8_l1(
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

    VXORPD(YMM(7), YMM(7), YMM(7)) //clear out registers
    VMOVAPD(YMM(24), YMM(7))
    VMOVAPD(YMM(25), YMM(7))
    VMOVAPD(YMM(26), YMM(7))
    VMOVAPD(YMM(27), YMM(7))
    VMOVAPD(YMM(28), YMM(7))
    VMOVAPD(YMM(29), YMM(7))
    VMOVAPD(YMM(30), YMM(7))
    VMOVAPD(YMM(31), YMM(7))

    MOV(RSI, VAR(k)) //loop index
    MOV(RAX, VAR(a)) //load address of a
    MOV(RBX, VAR(b)) //load address of b
    MOV(RCX, VAR(c)) //load address of c
    VMOVAPD(ZMM(0), MEM(RBX)) //pre-load b
    MOV(R12, VAR(rs_c))      //rs_c
    LEA(R13, MEM(R12,R12,2)) //*3
    LEA(R14, MEM(R12,R12,4)) //*5
    LEA(R15, MEM(R14,R12,2)) //*7
    MOV(R8, IMM(8*8)) //mr*sizeof(double)
    MOV(R9, IMM(8*8)) //nr*sizeof(double)
    LEA(RBX, MEM(RBX,R9,1)) //adjust b for pre-load

    TEST(RSI, RSI)
    JZ(POSTACCUM)

    PREFETCH_C_L1

    MOV(RDI, RSI)
    AND(RSI, IMM(7))
    SAR(RDI, IMM(3))
    JZ(TAIL_LOOP)

    LOOP_ALIGN
    LABEL(MAIN_LOOP)

        SUBITER(0)
        SUBITER(1)
        SUBITER(2)
        SUBITER(3)
        SUBITER(4)
        SUBITER(5)
        SUBITER(6)
        SUBITER(7)

        LEA(RAX, MEM(RAX,R8,8))
        LEA(RBX, MEM(RBX,R9,8))

        DEC(RDI)

    JNZ(MAIN_LOOP)

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

    MOV(RAX, VAR(rs_c))
    LEA(RAX, MEM(,RAX,8))
    LEA(RDX, MEM(RAX,RAX,2))
    MOV(RBX, VAR(cs_c))

    // Check if C is row stride. If not, jump to the slow scattered update
    CMP(RBX, IMM(1))
    JNE(SCATTEREDUPDATE)

        VCOMISD(XMM(1), XMM(7))
        JE(COLSTORBZ)

            UPDATE_C(24,25,26,27)
            UPDATE_C(28,29,30,31)

        JMP(END)
        LABEL(COLSTORBZ)

            UPDATE_C_BZ(24,25,26,27)
            UPDATE_C_BZ(28,29,30,31)

    JMP(END)
    LABEL(SCATTEREDUPDATE)

        MOV(RDI, VAR(offsetPtr))
        VMOVDQA64(ZMM(2), MEM(RDI,0*64))
        VPBROADCASTQ(ZMM(6), RBX)
        VPMULLQ(ZMM(2), ZMM(6), ZMM(2))

        VCOMISD(XMM(1), XMM(7))
        JE(SCATTERBZ)

            UPDATE_C_ROW_SCATTERED(24,25,26,27)
            UPDATE_C_ROW_SCATTERED(28,29,30,31)

        JMP(END)
        LABEL(SCATTERBZ)

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
