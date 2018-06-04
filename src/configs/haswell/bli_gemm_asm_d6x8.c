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
      derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#include "util/asm_x86.h"

#include "../ambi.hpp"

#define SGEMM_GS(n) \
    VMOVAPS(XMM(0), XMM(n)) \
    VFMADD231SS(XMM(0), XMM(1), MEM(RCX)) \
    VMOVSS(MEM(RCX), XMM(0)) \
    \
    VUNPCKHPS(XMM(0), XMM(0), XMM(0)) \
    VFMADD231SS(XMM(0), XMM(1), MEM(RCX,RSI,1)) \
    VMOVSS(MEM(RCX,RSI,1), XMM(0)) \
    \
    VUNPCKHPD(XMM(0), XMM(n), XMM(n)) \
    VFMADD231SS(XMM(0), XMM(1), MEM(RCX,RSI,2)) \
    VMOVSS(MEM(RCX,RSI,2), XMM(0)) \
    \
    VUNPCKHPS(XMM(0), XMM(0), XMM(0)) \
    VFMADD231SS(XMM(0), XMM(1), MEM(RCX,R13,1)) \
    VMOVSS(MEM(RCX,R13,1), XMM(0)) \
    \
    VEXTRACTF128(XMM(2), YMM(n), IMM(1)) \
    VMOVAPS(XMM(2), XMM(0)) \
    VFMADD231SS(XMM(0), XMM(1), MEM(RCX,RSI,4)) \
    VMOVSS(MEM(RCX,RSI,4), XMM(0)) \
    \
    VUNPCKHPS(XMM(0), XMM(0), XMM(0)) \
    VFMADD231SS(XMM(0), XMM(1), MEM(RCX,R15,1)) \
    VMOVSS(MEM(RCX,R15,1), XMM(0)) \
    \
    VUNPCKHPD(XMM(0), XMM(2), XMM(2)) \
    VFMADD231SS(XMM(0), XMM(1), MEM(RCX,R13,2)) \
    VMOVSS(MEM(RCX,R13,2), XMM(0)) \
    \
    VUNPCKHPS(XMM(0), XMM(0), XMM(0)) \
    VFMADD231SS(XMM(0), XMM(1), MEM(RCX,R10,1)) \
    VMOVSS(MEM(RCX,R10,1), XMM(0))

#define SGEMM_GS_BZ(n) \
    VMOVAPS(XMM(0), XMM(n)) \
    VMOVSS(MEM(RCX), XMM(0)) \
    \
    VUNPCKHPS(XMM(0), XMM(0), XMM(0)) \
    VMOVSS(MEM(RCX,RSI,1), XMM(0)) \
    \
    VUNPCKHPD(XMM(0), XMM(n), XMM(n)) \
    VMOVSS(MEM(RCX,RSI,2), XMM(0)) \
    \
    VUNPCKHPS(XMM(0), XMM(0), XMM(0)) \
    VMOVSS(MEM(RCX,R13,1), XMM(0)) \
    \
    VEXTRACTF128(XMM(2), YMM(n), IMM(1)) \
    VMOVAPS(XMM(2), XMM(0)) \
    VMOVSS(MEM(RCX,RSI,4), XMM(0)) \
    \
    VUNPCKHPS(XMM(0), XMM(0), XMM(0)) \
    VMOVSS(MEM(RCX,R15,1), XMM(0)) \
    \
    VUNPCKHPD(XMM(0), XMM(2), XMM(2)) \
    VMOVSS(MEM(RCX,R13,2), XMM(0)) \
    \
    VUNPCKHPS(XMM(0), XMM(0), XMM(0)) \
    VMOVSS(MEM(RCX,R10,1), XMM(0))

#define SGEMM_ITERATION(n) \
    \
    VBROADCASTSS(YMM(2), MEM(RAX,4*(0+6*n))) \
    VBROADCASTSS(YMM(3), MEM(RAX,4*(1+6*n))) \
    VFMADD231PS(YMM(4), YMM(2), YMM(0)) \
    VFMADD231PS(YMM(5), YMM(2), YMM(1)) \
    VFMADD231PS(YMM(6), YMM(3), YMM(0)) \
    VFMADD231PS(YMM(7), YMM(3), YMM(1)) \
    \
    VBROADCASTSS(YMM(2), MEM(RAX,4*(2+6*n))) \
    VBROADCASTSS(YMM(3), MEM(RAX,4*(3+6*n))) \
    VFMADD231PS(YMM(8), YMM(2), YMM(0)) \
    VFMADD231PS(YMM(9), YMM(2), YMM(1)) \
    VFMADD231PS(YMM(10), YMM(3), YMM(0)) \
    VFMADD231PS(YMM(11), YMM(3), YMM(1)) \
    \
    VBROADCASTSS(YMM(2), MEM(RAX,4*(4+6*n))) \
    VBROADCASTSS(YMM(3), MEM(RAX,4*(5+6*n))) \
    VFMADD231PS(YMM(12), YMM(2), YMM(0)) \
    VFMADD231PS(YMM(13), YMM(2), YMM(1)) \
    VFMADD231PS(YMM(14), YMM(3), YMM(0)) \
    VFMADD231PS(YMM(15), YMM(3), YMM(1)) \
    \
    VMOVAPS(YMM(0), MEM(RBX,32*(-4+2*n))) \
    VMOVAPS(YMM(1), MEM(RBX,32*(-3+2*n)))

void bli_sgemm_asm_6x16
     (
       dim_t               k_,
       float*     restrict alpha,
       float*     restrict a,
       float*     restrict b,
       float*     restrict beta,
       float*     restrict c, inc_t rs_c_, inc_t cs_c_,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

    int64_t k = k_;
    int64_t rs_c = rs_c_;
    int64_t cs_c = cs_c_;

	BEGIN_ASM

	VZEROALL()

	MOV(RAX, VAR(a))
	MOV(RBX, VAR(b))

    VMOVAPS(YMM(0), MEM(RBX,0*32)) // initialize loop by pre-loading
    VMOVAPS(YMM(1), MEM(RBX,1*32))

    ADD(RBX, IMM(32*6))

    MOV(RCX, VAR(c))
    MOV(RDI, VAR(rs_c))
    MOV(RSI, VAR(cs_c))
    LEA(RDI, MEM(,RDI,4)) // rs_c *= sizeof(float)
    LEA(RSI, MEM(,RSI,4)) // cs_c *= sizeof(float)

    CMP(RSI, IMM(4))
    JNE(.SCOLSTORPF)

        LEA(R13, MEM(RDI,RDI,2)) // r13 = 3*rs_c
        LEA(RDX, MEM(RCX,R13,1)) // rdx = c + 3*rs_c;

        PREFETCH(0, MEM(RCX      ))
        PREFETCH(0, MEM(RCX,RDI,1))
        PREFETCH(0, MEM(RCX,RDI,2))
        PREFETCH(0, MEM(RDX      ))
        PREFETCH(0, MEM(RDX,RDI,1))
        PREFETCH(0, MEM(RDX,RDI,2))
        PREFETCH(1, MEM(RCX,      64))
        PREFETCH(1, MEM(RCX,RDI,1,64))
        PREFETCH(1, MEM(RCX,RDI,2,64))
        PREFETCH(1, MEM(RDX,      64))
        PREFETCH(1, MEM(RDX,RDI,1,64))
        PREFETCH(1, MEM(RDX,RDI,2,64))

    JMP(.SACCUM)
    LABEL(.SCOLSTORPF)

        LEA(R13, MEM(RSI,RSI,2)) // r13 = 3*cs_c
        MOV(RDX, RCX)

        PREFETCH(0, MEM(RDX      ))
        PREFETCH(0, MEM(RDX,RSI,1))
        PREFETCH(0, MEM(RDX,RSI,2))
        PREFETCH(0, MEM(RDX,R13,1))
        LEA(RDX, MEM(RDX,RSI,4))
        PREFETCH(0, MEM(RDX      ))
        PREFETCH(0, MEM(RDX,RSI,1))
        PREFETCH(0, MEM(RDX,RSI,2))
        PREFETCH(0, MEM(RDX,R13,1))
        LEA(RDX, MEM(RDX,RSI,4))
        PREFETCH(0, MEM(RDX      ))
        PREFETCH(0, MEM(RDX,RSI,1))
        PREFETCH(0, MEM(RDX,RSI,2))
        PREFETCH(0, MEM(RDX,R13,1))
        LEA(RDX, MEM(RDX,RSI,4))
        PREFETCH(0, MEM(RDX      ))
        PREFETCH(0, MEM(RDX,RSI,1))
        PREFETCH(0, MEM(RDX,RSI,2))
        PREFETCH(0, MEM(RDX,R13,1))
        LEA(RDX, MEM(RDX,RSI,4))
        PREFETCH(1, MEM(RDX      ))
        PREFETCH(1, MEM(RDX,RSI,1))
        PREFETCH(1, MEM(RDX,RSI,2))
        PREFETCH(1, MEM(RDX,R13,1))
        LEA(RDX, MEM(RDX,RSI,4))
        PREFETCH(1, MEM(RDX      ))
        PREFETCH(1, MEM(RDX,RSI,1))
        PREFETCH(1, MEM(RDX,RSI,2))
        PREFETCH(1, MEM(RDX,R13,1))
        LEA(RDX, MEM(RDX,RSI,4))
        PREFETCH(1, MEM(RDX      ))
        PREFETCH(1, MEM(RDX,RSI,1))
        PREFETCH(1, MEM(RDX,RSI,2))
        PREFETCH(1, MEM(RDX,R13,1))
        LEA(RDX, MEM(RDX,RSI,4))
        PREFETCH(1, MEM(RDX      ))
        PREFETCH(1, MEM(RDX,RSI,1))
        PREFETCH(1, MEM(RDX,RSI,2))
        PREFETCH(1, MEM(RDX,R13,1))

    LABEL(.SACCUM)

    MOV(RSI, VAR(k))
    MOV(R8, RSI)
    SAR(RSI, IMM(2))
    JZ(.SCONSIDKLEFT)

    LABEL(.SLOOPKITER)

        PREFETCH(0, MEM(RAX,64*4))

        SGEMM_ITERATION(0)
        SGEMM_ITERATION(1)

        PREFETCH(0, MEM(RAX,76*4))

        SGEMM_ITERATION(2)
        SGEMM_ITERATION(3)

        ADD(RAX, IMM(4* 6*4))
        ADD(RBX, IMM(4*16*4))

        DEC(RSI)

    JNZ(.SLOOPKITER)

    LABEL(.SCONSIDKLEFT)

    MOV(RSI, R8)
    AND(RSI, IMM(3))
    JZ(.SPOSTACCUM)

    LABEL(.SLOOPKLEFT)

        PREFETCH(0, MEM(RAX,64*4))

        SGEMM_ITERATION(0)

        ADD(RAX, IMM(1* 6*4))
        ADD(RBX, IMM(1*16*4))

        DEC(RSI)

    JNZ(.SLOOPKLEFT)

	LABEL(.SPOSTACCUM)

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSS(YMM(0), MEM(RAX))
    VBROADCASTSS(YMM(1), MEM(RBX))

    VMULPS(YMM( 4), YMM( 4), YMM(0))
    VMULPS(YMM( 5), YMM( 5), YMM(0))
    VMULPS(YMM( 6), YMM( 6), YMM(0))
    VMULPS(YMM( 7), YMM( 7), YMM(0))
    VMULPS(YMM( 8), YMM( 8), YMM(0))
    VMULPS(YMM( 9), YMM( 9), YMM(0))
    VMULPS(YMM(10), YMM(10), YMM(0))
    VMULPS(YMM(11), YMM(11), YMM(0))
    VMULPS(YMM(12), YMM(12), YMM(0))
    VMULPS(YMM(13), YMM(13), YMM(0))
    VMULPS(YMM(14), YMM(14), YMM(0))
    VMULPS(YMM(15), YMM(15), YMM(0))

    MOV(RSI, VAR(cs_c))
    LEA(RSI, MEM(,RSI,4))

    LEA(RDX, MEM(RCX,RSI,8))

    LEA(R13, MEM(RSI,RSI,2))
    LEA(R15, MEM(RSI,RSI,4))
    LEA(R10, MEM(R13,RSI,4))

    VXORPS(YMM(0), YMM(0), YMM(0))
    VUCOMISS(XMM(1), XMM(0))
    JZ(.SBETAZERO)

        CMP(RSI, IMM(4))
        JE(.SROWSTORED)

        CMP(RDI, IMM(4))
        JE(.SCOLSTORED)

        LABEL(.SGENSTORED)

            SGEMM_GS(4)
            ADD(RCX, RDI)
            SGEMM_GS(6)
            ADD(RCX, RDI)
            SGEMM_GS(8)
            ADD(RCX, RDI)
            SGEMM_GS(10)
            ADD(RCX, RDI)
            SGEMM_GS(12)
            ADD(RCX, RDI)
            SGEMM_GS(14)
            MOV(RCX, RDX)
            SGEMM_GS(5)
            ADD(RCX, RDI)
            SGEMM_GS(7)
            ADD(RCX, RDI)
            SGEMM_GS(9)
            ADD(RCX, RDI)
            SGEMM_GS(11)
            ADD(RCX, RDI)
            SGEMM_GS(13)
            ADD(RCX, RDI)
            SGEMM_GS(15)

        JMP(.SDONE)
        LABEL(.SROWSTORED)

            VFMADD231PS(YMM(4), YMM(1), MEM(RCX))
            VFMADD231PS(YMM(5), YMM(1), MEM(RDX))
            VMOVUPS(MEM(RCX), YMM(4))
            VMOVUPS(MEM(RDX), YMM(5))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VFMADD231PS(YMM(6), YMM(1), MEM(RCX))
            VFMADD231PS(YMM(7), YMM(1), MEM(RDX))
            VMOVUPS(MEM(RCX), YMM(6))
            VMOVUPS(MEM(RDX), YMM(7))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VFMADD231PS(YMM(8), YMM(1), MEM(RCX))
            VFMADD231PS(YMM(9), YMM(1), MEM(RDX))
            VMOVUPS(MEM(RCX), YMM(8))
            VMOVUPS(MEM(RDX), YMM(9))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VFMADD231PS(YMM(10), YMM(1), MEM(RCX))
            VFMADD231PS(YMM(11), YMM(1), MEM(RDX))
            VMOVUPS(MEM(RCX), YMM(10))
            VMOVUPS(MEM(RDX), YMM(11))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VFMADD231PS(YMM(12), YMM(1), MEM(RCX))
            VFMADD231PS(YMM(13), YMM(1), MEM(RDX))
            VMOVUPS(MEM(RCX), YMM(12))
            VMOVUPS(MEM(RDX), YMM(13))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VFMADD231PS(YMM(14), YMM(1), MEM(RCX))
            VFMADD231PS(YMM(15), YMM(1), MEM(RDX))
            VMOVUPS(MEM(RCX), YMM(14))
            VMOVUPS(MEM(RDX), YMM(15))

        JMP(.SDONE)
        LABEL(.SCOLSTORED)

            //
            // Transpose and write out in four quadrants:
            //
            // +-------------------------------+-------------------------------+
            // |c00-c01-c02-c03-c04-c05-c06-c07|c08-c09-c0a-c0b-c0c-c0d-c0e-c0f|
            // |                               |                               |     +------+------+
            // |c10-c11-c12-c13-c14-c15-c16-c17|c18-c19-c1a-c1b-c1c-c1d-c1e-c1f|     |      |      |
            // |                               |                               |     |  Q1  |  Q3  |
            // |c20-c21-c22-c23-c24-c25-c26-c27|c28-c29-c2a-c2b-c2c-c2d-c2e-c2f|     |      |      |
            // |                               |                               |  =  +------+------+
            // |c30-c31-c32-c33-c34-c35-c36-c37|c38-c39-c3a-c3b-c3c-c3d-c3e-c3f|     |      |      |
            // +-------------------------------+-------------------------------+     |  Q2  |  Q4  |
            // |c40-c41-c42-c43-c44-c45-c46-c47|c48-c49-c4a-c4b-c4c-c4d-c4e-c4f|     |      |      |
            // |                               |                               |     +------+------+
            // |c50-c51-c52-c53-c54-c55-c56-c57|c58-c59-c5a-c5b-c5c-c5d-c5e-c5f|
            // +-------------------------------+-------------------------------+
            //
            //                                ||
            //                                \/
            //
            // +-------------------------------+-------------------------------+
            // |c00 c01 c02 c03 c04 c05 c06 c07|c08 c09 c0a c0b c0c c0d c0e c0f|
            // | |   |   |   |   |   |   |   | | |   |   |   |   |   |   |   | |
            // |c10 c11 c12 c13 c14 c15 c16 c17|c18 c19 c1a c1b c1c c1d c1e c1f|
            // | |   |   |   |   |   |   |   | | |   |   |   |   |   |   |   | |
            // |c20 c21 c22 c23 c24 c25 c26 c27|c28 c29 c2a c2b c2c c2d c2e c2f|
            // | |   |   |   |   |   |   |   | | |   |   |   |   |   |   |   | |
            // |c30 c31 c32 c33 c34 c35 c36 c37|c38 c39 c3a c3b c3c c3d c3e c3f|
            // +-------------------------------+-------------------------------+
            // |c40 c41 c42 c43 c44 c45 c46 c47|c48 c49 c4a c4b c4c c4d c4e c4f|
            // | |   |   |   |   |   |   |   | | |   |   |   |   |   |   |   | |
            // |c50 c51 c52 c53 c54 c55 c56 c57|c58 c59 c5a c5b c5c c5d c5e c5f|
            // +-------------------------------+-------------------------------+
            //

            VUNPCKLPS(YMM( 2), YMM(4), YMM( 6))
            VUNPCKHPS(YMM( 3), YMM(4), YMM( 6))
            VUNPCKLPS(YMM( 6), YMM(8), YMM(10))
            VUNPCKHPS(YMM(10), YMM(8), YMM(10))
            VSHUFPS (YMM( 0), YMM(2), YMM( 6), IMM(0x4e))
            VBLENDPS(YMM( 4), YMM(2), YMM( 0), IMM(0xcc))
            VBLENDPS(YMM( 6), YMM(0), YMM( 6), IMM(0xcc))
            VSHUFPS (YMM( 0), YMM(3), YMM(10), IMM(0x4e))
            VBLENDPS(YMM( 8), YMM(3), YMM( 0), IMM(0xcc))
            VBLENDPS(YMM(10), YMM(0), YMM(10), IMM(0xcc))
            VMOVUPS(XMM(2), MEM(RCX,     ))
            VMOVUPS(XMM(3), MEM(RCX,RSI,1))
            VINSERTF128(YMM(2), YMM(2), MEM(RCX,RSI,4), IMM(1))
            VINSERTF128(YMM(3), YMM(3), MEM(RCX,R15,1), IMM(1))
            VFMADD231PS(YMM(4), YMM(1), YMM(2))
            VFMADD231PS(YMM(6), YMM(1), YMM(3))
            VMOVUPS(MEM(RCX,     ), XMM(4))
            VMOVUPS(MEM(RCX,RSI,1), XMM(6))
            VEXTRACTF128(MEM(RCX,RSI,4), YMM(4), IMM(1))
            VEXTRACTF128(MEM(RCX,R15,1), YMM(6), IMM(1))
            VMOVUPS(XMM(2), MEM(RCX,RSI,2))
            VMOVUPS(XMM(3), MEM(RCX,R13,1))
            VINSERTF128(YMM(2), YMM(2), MEM(RCX,R13,2), IMM(1))
            VINSERTF128(YMM(3), YMM(3), MEM(RCX,R10,1), IMM(1))
            VFMADD231PS(YMM( 8), YMM(1), YMM(2))
            VFMADD231PS(YMM(10), YMM(1), YMM(3))
            VMOVUPS(MEM(RCX,RSI,2), XMM( 8))
            VMOVUPS(MEM(RCX,R13,1), XMM(10))
            VEXTRACTF128(MEM(RCX,R13,2), YMM( 8), IMM(1))
            VEXTRACTF128(MEM(RCX,R10,1), YMM(10), IMM(1))

            VUNPCKLPS(YMM(2), YMM(12), YMM(14))
            VUNPCKHPS(YMM(3), YMM(12), YMM(14))
            VMOVLPS(XMM(12), XMM(12), MEM(RCX,      16))
            VMOVHPS(XMM(12), XMM(12), MEM(RCX,RSI,1,16))
            VMOVLPS(XMM(0), XMM(0), MEM(RCX,RSI,4,16))
            VMOVHPS(XMM(0), XMM(0), MEM(RCX,R15,1,16))
            VINSERTF128(YMM(12), YMM(12), XMM(0), IMM(1))
            VMOVLPS(XMM(14), XMM(14), MEM(RCX,RSI,2,16))
            VMOVHPS(XMM(14), XMM(14), MEM(RCX,R13,1,16))
            VMOVLPS(XMM(0), XMM(0), MEM(RCX,R13,2,16))
            VMOVHPS(XMM(0), XMM(0), MEM(RCX,R10,1,16))
            VINSERTF128(YMM(14), YMM(14), XMM(0), IMM(1))
            VFMADD231PS(YMM(2), YMM(1), YMM(12))
            VFMADD231PS(YMM(3), YMM(1), YMM(14))
            VMOVLPS(MEM(RCX,      16), XMM(2))
            VMOVHPS(MEM(RCX,RSI,1,16), XMM(2))
            VMOVLPS(MEM(RCX,RSI,2,16), XMM(3))
            VMOVHPS(MEM(RCX,R13,1,16), XMM(3))
            VEXTRACTF128(XMM(2), YMM(2), IMM(1))
            VEXTRACTF128(XMM(3), YMM(3), IMM(1))
            VMOVLPS(MEM(RCX,RSI,4,16), XMM(2))
            VMOVHPS(MEM(RCX,R15,1,16), XMM(2))
            VMOVLPS(MEM(RCX,R13,2,16), XMM(3))
            VMOVHPS(MEM(RCX,R10,1,16), XMM(3))

            VUNPCKLPS(YMM( 2), YMM(5), YMM( 7))
            VUNPCKHPS(YMM( 3), YMM(5), YMM( 7))
            VUNPCKLPS(YMM( 7), YMM(9), YMM(11))
            VUNPCKHPS(YMM(11), YMM(9), YMM(11))
            VSHUFPS (YMM( 0), YMM(2), YMM( 7), IMM(0x4e))
            VBLENDPS(YMM( 5), YMM(2), YMM( 0), IMM(0xcc))
            VBLENDPS(YMM( 7), YMM(0), YMM( 7), IMM(0xcc))
            VSHUFPS (YMM( 0), YMM(3), YMM(11), IMM(0x4e))
            VBLENDPS(YMM( 9), YMM(3), YMM( 0), IMM(0xcc))
            VBLENDPS(YMM(11), YMM(0), YMM(11), IMM(0xcc))
            VMOVUPS(XMM(2), MEM(RDX,     ))
            VMOVUPS(XMM(3), MEM(RDX,RSI,1))
            VINSERTF128(YMM(2), YMM(2), MEM(RDX,RSI,4), IMM(1))
            VINSERTF128(YMM(3), YMM(3), MEM(RDX,R15,1), IMM(1))
            VFMADD231PS(YMM(5), YMM(1), YMM(2))
            VFMADD231PS(YMM(7), YMM(1), YMM(3))
            VMOVUPS(MEM(RDX,     ), XMM(5))
            VMOVUPS(MEM(RDX,RSI,1), XMM(7))
            VEXTRACTF128(MEM(RDX,RSI,4), YMM(5), IMM(1))
            VEXTRACTF128(MEM(RDX,R15,1), YMM(7), IMM(1))
            VMOVUPS(XMM(2), MEM(RDX,RSI,2))
            VMOVUPS(XMM(3), MEM(RDX,R13,1))
            VINSERTF128(YMM(2), YMM(2), MEM(RDX,R13,2), IMM(1))
            VINSERTF128(YMM(3), YMM(3), MEM(RDX,R10,1), IMM(1))
            VFMADD231PS(YMM( 9), YMM(1), YMM(2))
            VFMADD231PS(YMM(11), YMM(1), YMM(3))
            VMOVUPS(MEM(RDX,RSI,2), XMM( 9))
            VMOVUPS(MEM(RDX,R13,1), XMM(11))
            VEXTRACTF128(MEM(RDX,R13,2), YMM( 9), IMM(1))
            VEXTRACTF128(MEM(RDX,R10,1), YMM(11), IMM(1))

            VUNPCKLPS(YMM(2), YMM(13), YMM(15))
            VUNPCKHPS(YMM(3), YMM(13), YMM(15))
            VMOVLPS(XMM(13), XMM(13), MEM(RDX,      16))
            VMOVHPS(XMM(13), XMM(13), MEM(RDX,RSI,1,16))
            VMOVLPS(XMM(0), XMM(0), MEM(RDX,RSI,4,16))
            VMOVHPS(XMM(0), XMM(0), MEM(RDX,R15,1,16))
            VINSERTF128(YMM(13), YMM(13), XMM(0), IMM(1))
            VMOVLPS(XMM(15), XMM(15), MEM(RDX,RSI,2,16))
            VMOVHPS(XMM(15), XMM(15), MEM(RDX,R13,1,16))
            VMOVLPS(XMM(0), XMM(0), MEM(RDX,R13,2,16))
            VMOVHPS(XMM(0), XMM(0), MEM(RDX,R10,1,16))
            VINSERTF128(YMM(15), YMM(15), XMM(0), IMM(1))
            VFMADD231PS(YMM(2), YMM(1), YMM(13))
            VFMADD231PS(YMM(3), YMM(1), YMM(15))
            VMOVLPS(MEM(RDX,      16), XMM(2))
            VMOVHPS(MEM(RDX,RSI,1,16), XMM(2))
            VMOVLPS(MEM(RDX,RSI,2,16), XMM(3))
            VMOVHPS(MEM(RDX,R13,1,16), XMM(3))
            VEXTRACTF128(XMM(2), YMM(2), IMM(1))
            VEXTRACTF128(XMM(3), YMM(3), IMM(1))
            VMOVLPS(MEM(RDX,RSI,4,16), XMM(2))
            VMOVHPS(MEM(RDX,R15,1,16), XMM(2))
            VMOVLPS(MEM(RDX,R13,2,16), XMM(3))
            VMOVHPS(MEM(RDX,R10,1,16), XMM(3))

	JMP(.SDONE)
	LABEL(.SBETAZERO)

        CMP(RSI, IMM(4))
        JE(.SROWSTORBZ)

        CMP(RDI, IMM(4))
        JE(.SCOLSTORBZ)

        LABEL(.SGENSTORBZ)

            SGEMM_GS_BZ(4)
            ADD(RCX, RDI)
            SGEMM_GS_BZ(6)
            ADD(RCX, RDI)
            SGEMM_GS_BZ(8)
            ADD(RCX, RDI)
            SGEMM_GS_BZ(10)
            ADD(RCX, RDI)
            SGEMM_GS_BZ(12)
            ADD(RCX, RDI)
            SGEMM_GS_BZ(14)
            MOV(RCX, RDX)
            SGEMM_GS_BZ(5)
            ADD(RCX, RDI)
            SGEMM_GS_BZ(7)
            ADD(RCX, RDI)
            SGEMM_GS_BZ(9)
            ADD(RCX, RDI)
            SGEMM_GS_BZ(11)
            ADD(RCX, RDI)
            SGEMM_GS_BZ(13)
            ADD(RCX, RDI)
            SGEMM_GS_BZ(15)

        JMP(.SDONE)
        LABEL(.SROWSTORBZ)

            VMOVUPS(MEM(RCX), YMM(4))
            VMOVUPS(MEM(RDX), YMM(5))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VMOVUPS(MEM(RCX), YMM(6))
            VMOVUPS(MEM(RDX), YMM(7))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VMOVUPS(MEM(RCX), YMM(8))
            VMOVUPS(MEM(RDX), YMM(9))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VMOVUPS(MEM(RCX), YMM(10))
            VMOVUPS(MEM(RDX), YMM(11))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VMOVUPS(MEM(RCX), YMM(12))
            VMOVUPS(MEM(RDX), YMM(13))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VMOVUPS(MEM(RCX), YMM(14))
            VMOVUPS(MEM(RDX), YMM(15))

        JMP(.SDONE)
        LABEL(.SCOLSTORBZ)

            //
            // Transpose and write out in four quadrants:
            //
            // +-------------------------------+-------------------------------+
            // |c00-c01-c02-c03-c04-c05-c06-c07|c08-c09-c0a-c0b-c0c-c0d-c0e-c0f|
            // |                               |                               |     +------+------+
            // |c10-c11-c12-c13-c14-c15-c16-c17|c18-c19-c1a-c1b-c1c-c1d-c1e-c1f|     |      |      |
            // |                               |                               |     |  Q1  |  Q3  |
            // |c20-c21-c22-c23-c24-c25-c26-c27|c28-c29-c2a-c2b-c2c-c2d-c2e-c2f|     |      |      |
            // |                               |                               |  =  +------+------+
            // |c30-c31-c32-c33-c34-c35-c36-c37|c38-c39-c3a-c3b-c3c-c3d-c3e-c3f|     |      |      |
            // +-------------------------------+-------------------------------+     |  Q2  |  Q4  |
            // |c40-c41-c42-c43-c44-c45-c46-c47|c48-c49-c4a-c4b-c4c-c4d-c4e-c4f|     |      |      |
            // |                               |                               |     +------+------+
            // |c50-c51-c52-c53-c54-c55-c56-c57|c58-c59-c5a-c5b-c5c-c5d-c5e-c5f|
            // +-------------------------------+-------------------------------+
            //
            //                                ||
            //                                \/
            //
            // +-------------------------------+-------------------------------+
            // |c00 c01 c02 c03 c04 c05 c06 c07|c08 c09 c0a c0b c0c c0d c0e c0f|
            // | |   |   |   |   |   |   |   | | |   |   |   |   |   |   |   | |
            // |c10 c11 c12 c13 c14 c15 c16 c17|c18 c19 c1a c1b c1c c1d c1e c1f|
            // | |   |   |   |   |   |   |   | | |   |   |   |   |   |   |   | |
            // |c20 c21 c22 c23 c24 c25 c26 c27|c28 c29 c2a c2b c2c c2d c2e c2f|
            // | |   |   |   |   |   |   |   | | |   |   |   |   |   |   |   | |
            // |c30 c31 c32 c33 c34 c35 c36 c37|c38 c39 c3a c3b c3c c3d c3e c3f|
            // +-------------------------------+-------------------------------+
            // |c40 c41 c42 c43 c44 c45 c46 c47|c48 c49 c4a c4b c4c c4d c4e c4f|
            // | |   |   |   |   |   |   |   | | |   |   |   |   |   |   |   | |
            // |c50 c51 c52 c53 c54 c55 c56 c57|c58 c59 c5a c5b c5c c5d c5e c5f|
            // +-------------------------------+-------------------------------+
            //

            VUNPCKLPS(YMM( 2), YMM(4), YMM( 6))
            VUNPCKHPS(YMM( 3), YMM(4), YMM( 6))
            VUNPCKLPS(YMM( 6), YMM(8), YMM(10))
            VUNPCKHPS(YMM(10), YMM(8), YMM(10))
            VSHUFPS (YMM( 0), YMM(2), YMM( 6), IMM(0x4e))
            VBLENDPS(YMM( 4), YMM(2), YMM( 0), IMM(0xcc))
            VBLENDPS(YMM( 6), YMM(0), YMM( 6), IMM(0xcc))
            VSHUFPS (YMM( 0), YMM(3), YMM(10), IMM(0x4e))
            VBLENDPS(YMM( 8), YMM(3), YMM( 0), IMM(0xcc))
            VBLENDPS(YMM(10), YMM(0), YMM(10), IMM(0xcc))
            VMOVUPS(MEM(RCX,     ), XMM(4))
            VMOVUPS(MEM(RCX,RSI,1), XMM(6))
            VEXTRACTF128(MEM(RCX,RSI,4), YMM(4), IMM(1))
            VEXTRACTF128(MEM(RCX,R15,1), YMM(6), IMM(1))
            VMOVUPS(MEM(RCX,RSI,2), XMM( 8))
            VMOVUPS(MEM(RCX,R13,1), XMM(10))
            VEXTRACTF128(MEM(RCX,R13,2), YMM( 8), IMM(1))
            VEXTRACTF128(MEM(RCX,R10,1), YMM(10), IMM(1))

            VUNPCKLPS(YMM(2), YMM(12), YMM(14))
            VUNPCKHPS(YMM(3), YMM(12), YMM(14))
            VMOVLPS(MEM(RCX,      16), XMM(2))
            VMOVHPS(MEM(RCX,RSI,1,16), XMM(2))
            VMOVLPS(MEM(RCX,RSI,2,16), XMM(3))
            VMOVHPS(MEM(RCX,R13,1,16), XMM(3))
            VEXTRACTF128(XMM(2), YMM(2), IMM(1))
            VEXTRACTF128(XMM(3), YMM(3), IMM(1))
            VMOVLPS(MEM(RCX,RSI,4,16), XMM(2))
            VMOVHPS(MEM(RCX,R15,1,16), XMM(2))
            VMOVLPS(MEM(RCX,R13,2,16), XMM(3))
            VMOVHPS(MEM(RCX,R10,1,16), XMM(3))

            VUNPCKLPS(YMM( 2), YMM(5), YMM( 7))
            VUNPCKHPS(YMM( 3), YMM(5), YMM( 7))
            VUNPCKLPS(YMM( 7), YMM(9), YMM(11))
            VUNPCKHPS(YMM(11), YMM(9), YMM(11))
            VSHUFPS (YMM( 0), YMM(2), YMM( 7), IMM(0x4e))
            VBLENDPS(YMM( 5), YMM(2), YMM( 0), IMM(0xcc))
            VBLENDPS(YMM( 7), YMM(0), YMM( 7), IMM(0xcc))
            VSHUFPS (YMM( 0), YMM(3), YMM(11), IMM(0x4e))
            VBLENDPS(YMM( 9), YMM(3), YMM( 0), IMM(0xcc))
            VBLENDPS(YMM(11), YMM(0), YMM(11), IMM(0xcc))
            VMOVUPS(MEM(RDX,     ), XMM(5))
            VMOVUPS(MEM(RDX,RSI,1), XMM(7))
            VEXTRACTF128(MEM(RDX,RSI,4), YMM(5), IMM(1))
            VEXTRACTF128(MEM(RDX,R15,1), YMM(7), IMM(1))
            VMOVUPS(MEM(RDX,RSI,2), XMM( 9))
            VMOVUPS(MEM(RDX,R13,1), XMM(11))
            VEXTRACTF128(MEM(RDX,R13,2), YMM( 9), IMM(1))
            VEXTRACTF128(MEM(RDX,R10,1), YMM(11), IMM(1))

            VUNPCKLPS(YMM(2), YMM(13), YMM(15))
            VUNPCKHPS(YMM(3), YMM(13), YMM(15))
            VMOVLPS(MEM(RDX,      16), XMM(2))
            VMOVHPS(MEM(RDX,RSI,1,16), XMM(2))
            VMOVLPS(MEM(RDX,RSI,2,16), XMM(3))
            VMOVHPS(MEM(RDX,R13,1,16), XMM(3))
            VEXTRACTF128(XMM(2), YMM(2), IMM(1))
            VEXTRACTF128(XMM(3), YMM(3), IMM(1))
            VMOVLPS(MEM(RDX,RSI,4,16), XMM(2))
            VMOVHPS(MEM(RDX,R15,1,16), XMM(2))
            VMOVLPS(MEM(RDX,R13,2,16), XMM(3))
            VMOVHPS(MEM(RDX,R10,1,16), XMM(3))

	LABEL(.SDONE)

	VZEROUPPER()

	END_ASM
	(
	    : // output operands (none)
        : // input operands
          [k]     "m" (k),
          [a]     "m" (a),
          [b]     "m" (b),
          [alpha] "m" (alpha),
          [beta]  "m" (beta),
          [c]     "m" (c),
          [rs_c]  "m" (rs_c),
          [cs_c]  "m" (cs_c)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
          "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
          "xmm0", "xmm1", "xmm2", "xmm3",
          "xmm4", "xmm5", "xmm6", "xmm7",
          "xmm8", "xmm9", "xmm10", "xmm11",
          "xmm12", "xmm13", "xmm14", "xmm15",
          "memory"
	)
}

#define DGEMM_GS(n) \
    VMOVAPD(XMM(0), XMM(n)) \
    VFMADD231SD(XMM(0), XMM(1), MEM(RCX)) \
    VMOVSD(MEM(RCX), XMM(0)) \
    \
    VUNPCKHPD(XMM(0), XMM(0), XMM(0)) \
    VFMADD231SD(XMM(0), XMM(1), MEM(RCX,RSI,1)) \
    VMOVSD(MEM(RCX,RSI,1), XMM(0)) \
    \
    VEXTRACTF128(XMM(2), YMM(n), IMM(1)) \
    VMOVAPD(XMM(2), XMM(0)) \
    VFMADD231SD(XMM(0), XMM(1), MEM(RCX,RSI,2)) \
    VMOVSD(MEM(RCX,RSI,2), XMM(0)) \
    \
    VUNPCKHPD(XMM(0), XMM(0), XMM(0)) \
    VFMADD231SD(XMM(0), XMM(1), MEM(RCX,R13,1)) \
    VMOVSD(MEM(RCX,R13,1), XMM(0))

#define DGEMM_GS_BZ(n) \
    VMOVAPD(XMM(0), XMM(n)) \
    VMOVSD(MEM(RCX), XMM(0)) \
    \
    VUNPCKHPD(XMM(0), XMM(0), XMM(0)) \
    VMOVSD(MEM(RCX,RSI,1), XMM(0)) \
    \
    VEXTRACTF128(XMM(2), YMM(n), IMM(1)) \
    VMOVAPD(XMM(2), XMM(0)) \
    VMOVSD(MEM(RCX,RSI,2), XMM(0)) \
    \
    VUNPCKHPD(XMM(0), XMM(0), XMM(0)) \
    VMOVSD(MEM(RCX,R13,1), XMM(0))

#define DGEMM_ITERATION(n) \
    VBROADCASTSD(YMM(2), MEM(RAX,8*(0+6*n))) \
    VBROADCASTSD(YMM(3), MEM(RAX,8*(1+6*n))) \
    VFMADD231PD(YMM(4), YMM(2), YMM(0)) \
    VFMADD231PD(YMM(5), YMM(2), YMM(1)) \
    VFMADD231PD(YMM(6), YMM(3), YMM(0)) \
    VFMADD231PD(YMM(7), YMM(3), YMM(1)) \
    \
    VBROADCASTSD(YMM(2), MEM(RAX,8*(2+6*n))) \
    VBROADCASTSD(YMM(3), MEM(RAX,8*(3+6*n))) \
    VFMADD231PD(YMM(8), YMM(2), YMM(0)) \
    VFMADD231PD(YMM(9), YMM(2), YMM(1)) \
    VFMADD231PD(YMM(10), YMM(3), YMM(0)) \
    VFMADD231PD(YMM(11), YMM(3), YMM(1)) \
    \
    VBROADCASTSD(YMM(2), MEM(RAX,8*(4+6*n))) \
    VBROADCASTSD(YMM(3), MEM(RAX,8*(5+6*n))) \
    VFMADD231PD(YMM(12), YMM(2), YMM(0)) \
    VFMADD231PD(YMM(13), YMM(2), YMM(1)) \
    VFMADD231PD(YMM(14), YMM(3), YMM(0)) \
    VFMADD231PD(YMM(15), YMM(3), YMM(1)) \
    \
    VMOVAPD(YMM(0), MEM(RBX,32*(-4+2*n))) \
    VMOVAPD(YMM(1), MEM(RBX,32*(-3+2*n)))

#define PREFETCH_C_6(level,C,s1,s3,s5,off) \
   PREFETCH(level, MEM(C,     off)) \
   PREFETCH(level, MEM(C,s1,1,off)) \
   PREFETCH(level, MEM(C,s1,2,off)) \
   PREFETCH(level, MEM(C,s3,1,off)) \
   PREFETCH(level, MEM(C,s1,4,off)) \
   PREFETCH(level, MEM(C,s5,1,off))

#define PREFETCH_C_8(level,C,s1,s3,s5,s7,off) \
   PREFETCH_C_6(level,C,s1,s3,s5,off) \
   PREFETCH(level, MEM(C,s3,2,off)) \
   PREFETCH(level, MEM(C,s7,1,off))

void bli_dgemm_asm_6x8
     (
       dim_t               k_,
       double*    restrict alpha,
       double*    restrict a,
       double*    restrict b,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c_, inc_t cs_c_,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    //void*   a_next = bli_auxinfo_next_a( data );
    //void*   b_next = bli_auxinfo_next_b( data );

    int64_t k = k_;
    int64_t rs_c = rs_c_;
    int64_t cs_c = cs_c_;

    BEGIN_ASM

    VZEROALL()

    MOV(RAX, VAR(a))
    MOV(RBX, VAR(b))

    VMOVAPS(YMM(0), MEM(RBX,0*32)) // initialize loop by pre-loading
    VMOVAPS(YMM(1), MEM(RBX,1*32))

    ADD(RBX, IMM(32*6))

    MOV(RCX, VAR(c))
    MOV(RDI, VAR(rs_c))
    MOV(RSI, VAR(cs_c))
    LEA(RDI, MEM(,RDI,8)) // rs_c *= sizeof(double)
    LEA(RSI, MEM(,RSI,8)) // cs_c *= sizeof(double)

    CMP(RSI, IMM(8))
    JNE(.DCOLSTORPF)

        LEA(R13, MEM(RDI,RDI,2)) // r13 = 3*rs_c
        LEA(R14, MEM(RDI,RDI,4)) // r14 = 5*rs_c

#if PF_C_L2 == PF_JR || PF_C_L2 == PF_NEAR
        PREFETCH_C_6(1,RCX,RDI,R13,R14,64)
#else
        LEA(RDX, MEM(RCX,R13,2)) // rdx = c + 6*rs_c;
        PREFETCH_C_6(1,RDX,RDI,R13,R14,0)
#endif

        PREFETCH_C_6(0,RCX,RDI,R13,R14,0)

    JMP(.DACCUM)
    LABEL(.DCOLSTORPF)

        LEA(R13, MEM(RSI,RSI,2)) // r13 = 3*cs_c
        LEA(R14, MEM(RSI,RSI,4)) // r14 = 5*cs_c
        LEA(R15, MEM(RSI,R13,2)) // r15 = 7*cs_c

#if PF_C_L2 == PF_IR || PF_C_L2 == PF_NEAR
        PREFETCH_C_8(1,RCX,RSI,R13,R14,R15,48)
#else
        LEA(RDX, MEM(RCX,RSI,8)) // rdx = c + 8*cs_c;
        PREFETCH_C_8(1,RDX,RSI,R13,R14,R15,0)
#endif

        PREFETCH_C_8(0,RCX,RSI,R13,R14,R15,0)

    LABEL(.DACCUM)

    MOV(RSI, VAR(k))
    MOV(R8, RSI)
    SAR(RSI, IMM(2))
    JZ(.DCONSIDKLEFT)

    LABEL(.DLOOPKITER)

        PREFETCH(0, MEM(RAX,64*8))

        DGEMM_ITERATION(0)

        PREFETCH(0, MEM(RAX,72*8))

        DGEMM_ITERATION(1)

        PREFETCH(0, MEM(RAX,80*8))

        DGEMM_ITERATION(2)
        DGEMM_ITERATION(3)

        ADD(RAX, IMM(4*6*8))
        ADD(RBX, IMM(4*8*8))

        DEC(RSI)

    JNZ(.DLOOPKITER)

    LABEL(.DCONSIDKLEFT)

    MOV(RSI, R8)
    AND(RSI, IMM(3))
    JZ(.DPOSTACCUM)

    LABEL(.DLOOPKLEFT)

        PREFETCH(0, MEM(RAX,64*8))

        DGEMM_ITERATION(0)

        ADD(RAX, IMM(1*6*8))
        ADD(RBX, IMM(1*8*8))

        DEC(RSI)

    JNZ(.DLOOPKLEFT)

    LABEL(.DPOSTACCUM)

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSD(YMM(0), MEM(RAX))
    VBROADCASTSD(YMM(1), MEM(RBX))

    VMULPD(YMM( 4), YMM( 4), YMM(0))
    VMULPD(YMM( 5), YMM( 5), YMM(0))
    VMULPD(YMM( 6), YMM( 6), YMM(0))
    VMULPD(YMM( 7), YMM( 7), YMM(0))
    VMULPD(YMM( 8), YMM( 8), YMM(0))
    VMULPD(YMM( 9), YMM( 9), YMM(0))
    VMULPD(YMM(10), YMM(10), YMM(0))
    VMULPD(YMM(11), YMM(11), YMM(0))
    VMULPD(YMM(12), YMM(12), YMM(0))
    VMULPD(YMM(13), YMM(13), YMM(0))
    VMULPD(YMM(14), YMM(14), YMM(0))
    VMULPD(YMM(15), YMM(15), YMM(0))

    MOV(RSI, VAR(cs_c))
    LEA(RSI, MEM(,RSI,8))

    LEA(RDX, MEM(RCX,RSI,4))

    LEA(R13, MEM(RSI,RSI,2))

    VXORPD(YMM(0), YMM(0), YMM(0))
    VUCOMISD(XMM(1), XMM(0))
    JZ(.DBETAZERO)

        CMP(RSI, IMM(8))
        JE(.DROWSTORED)

#if AMBI_METHOD != GS
        CMP(RDI, IMM(8))
        JE(.DCOLSTORED)
#endif

        LABEL(.DGENSTORED)

            DGEMM_GS(4)
            ADD(RCX, RDI)
            DGEMM_GS(6)
            ADD(RCX, RDI)
            DGEMM_GS(8)
            ADD(RCX, RDI)
            DGEMM_GS(10)
            ADD(RCX, RDI)
            DGEMM_GS(12)
            ADD(RCX, RDI)
            DGEMM_GS(14)
            MOV(RCX, RDX)
            DGEMM_GS(5)
            ADD(RCX, RDI)
            DGEMM_GS(7)
            ADD(RCX, RDI)
            DGEMM_GS(9)
            ADD(RCX, RDI)
            DGEMM_GS(11)
            ADD(RCX, RDI)
            DGEMM_GS(13)
            ADD(RCX, RDI)
            DGEMM_GS(15)

        JMP(.DDONE)
        LABEL(.DROWSTORED)

            VFMADD231PD(YMM(4), YMM(1), MEM(RCX))
            VFMADD231PD(YMM(5), YMM(1), MEM(RDX))
            VMOVUPD(MEM(RCX), YMM(4))
            VMOVUPD(MEM(RDX), YMM(5))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VFMADD231PD(YMM(6), YMM(1), MEM(RCX))
            VFMADD231PD(YMM(7), YMM(1), MEM(RDX))
            VMOVUPD(MEM(RCX), YMM(6))
            VMOVUPD(MEM(RDX), YMM(7))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VFMADD231PD(YMM(8), YMM(1), MEM(RCX))
            VFMADD231PD(YMM(9), YMM(1), MEM(RDX))
            VMOVUPD(MEM(RCX), YMM(8))
            VMOVUPD(MEM(RDX), YMM(9))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VFMADD231PD(YMM(10), YMM(1), MEM(RCX))
            VFMADD231PD(YMM(11), YMM(1), MEM(RDX))
            VMOVUPD(MEM(RCX), YMM(10))
            VMOVUPD(MEM(RDX), YMM(11))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VFMADD231PD(YMM(12), YMM(1), MEM(RCX))
            VFMADD231PD(YMM(13), YMM(1), MEM(RDX))
            VMOVUPD(MEM(RCX), YMM(12))
            VMOVUPD(MEM(RDX), YMM(13))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VFMADD231PD(YMM(14), YMM(1), MEM(RCX))
            VFMADD231PD(YMM(15), YMM(1), MEM(RDX))
            VMOVUPD(MEM(RCX), YMM(14))
            VMOVUPD(MEM(RDX), YMM(15))

        JMP(.DDONE)
        LABEL(.DCOLSTORED)

            //
            // Transpose and write out in four quadrants:
            //
            // +---------------+---------------+
            // |c00-c01-c02-c03|c04-c05-c06-c07|
            // |               |               |      +------+------+
            // |c10-c11-c12-c13|c14-c15-c16-c17|      |      |      |
            // |               |               |      |  Q1  |  Q3  |
            // |c20-c21-c22-c23|c24-c25-c26-c27|      |      |      |
            // |               |               |   =  +------+------+
            // |c30-c31-c32-c33|c34-c35-c36-c37|      |      |      |
            // +---------------+---------------+      |  Q2  |  Q4  |
            // |c40-c41-c42-c43|c44-c45-c46-c47|      |      |      |
            // |               |               |      +------+------+
            // |c50-c51-c52-c53|c54-c55-c56-c57|
            // +---------------+---------------+
            //
            //                ||
            //                \/
            //
            // +---------------+---------------+
            // |c00 c01 c02 c03|c04 c05 c06 c07|
            // | |   |   |   | | |   |   |   | |
            // |c10 c11 c12 c13|c14 c15 c16 c17|
            // | |   |   |   | | |   |   |   | |
            // |c20 c21 c22 c23|c24 c25 c26 c27|
            // | |   |   |   | | |   |   |   | |
            // |c30 c31 c32 c33|c34 c35 c36 c37|
            // +---------------+---------------+
            // |c40 c41 c42 c43|c44 c45 c46 c47|
            // | |   |   |   | | |   |   |   | |
            // |c50 c51 c52 c53|c54 c55 c56 c57|
            // +---------------+---------------+
            //

            VUNPCKLPD(YMM(2), YMM(4), YMM( 6))
            VUNPCKHPD(YMM(6), YMM(4), YMM( 6))
            VUNPCKHPD(YMM(3), YMM(8), YMM(10))
            VUNPCKLPD(YMM(8), YMM(8), YMM(10))
            VPERM2F128(YMM( 4), YMM(2), YMM(8), IMM(0x21))
            VPERM2F128(YMM(10), YMM(6), YMM(3), IMM(0x21))
            VBLENDPD(YMM( 8), YMM( 8), YMM( 4), IMM(0x3))
            VBLENDPD(YMM( 4), YMM( 4), YMM( 2), IMM(0x3))
            VBLENDPD(YMM( 6), YMM(10), YMM( 6), IMM(0x3))
            VBLENDPD(YMM(10), YMM( 3), YMM(10), IMM(0x3))
            VFMADD231PD(YMM( 4), YMM(1), MEM(RCX      ))
            VFMADD231PD(YMM( 6), YMM(1), MEM(RCX,RSI,1))
            VFMADD231PD(YMM( 8), YMM(1), MEM(RCX,RSI,2))
            VFMADD231PD(YMM(10), YMM(1), MEM(RCX,R13,1))
            VMOVUPD(MEM(RCX      ), YMM( 4))
            VMOVUPD(MEM(RCX,RSI,1), YMM( 6))
            VMOVUPD(MEM(RCX,RSI,2), YMM( 8))
            VMOVUPD(MEM(RCX,R13,1), YMM(10))

            VUNPCKLPD(YMM(2), YMM(12), YMM(14))
            VUNPCKHPD(YMM(3), YMM(12), YMM(14))
            VMOVUPD(XMM(12), MEM(RCX,      32))
            VMOVUPD(XMM(14), MEM(RCX,RSI,1,32))
            VINSERTF128(YMM(12), YMM(12), MEM(RCX,RSI,2,32), IMM(1))
            VINSERTF128(YMM(14), YMM(14), MEM(RCX,R13,1,32), IMM(1))
            VFMADD231PD(YMM(2), YMM(1), YMM(12))
            VFMADD231PD(YMM(3), YMM(1), YMM(14))
            VMOVUPD(MEM(RCX,      32), XMM(2))
            VMOVUPD(MEM(RCX,RSI,1,32), XMM(3))
            VEXTRACTF128(MEM(RCX,RSI,2,32), YMM(2), IMM(1))
            VEXTRACTF128(MEM(RCX,R13,1,32), YMM(3), IMM(1))

            VUNPCKLPD(YMM(2), YMM(5), YMM( 7))
            VUNPCKHPD(YMM(7), YMM(5), YMM( 7))
            VUNPCKHPD(YMM(3), YMM(9), YMM(11))
            VUNPCKLPD(YMM(9), YMM(9), YMM(11))
            VPERM2F128(YMM( 5), YMM(2), YMM(9), IMM(0x21))
            VPERM2F128(YMM(11), YMM(7), YMM(3), IMM(0x21))
            VBLENDPD(YMM( 9), YMM( 9), YMM( 5), IMM(0x3))
            VBLENDPD(YMM( 5), YMM( 5), YMM( 2), IMM(0x3))
            VBLENDPD(YMM( 7), YMM(11), YMM( 7), IMM(0x3))
            VBLENDPD(YMM(11), YMM( 3), YMM(11), IMM(0x3))
            VFMADD231PD(YMM( 5), YMM(1), MEM(RDX      ))
            VFMADD231PD(YMM( 7), YMM(1), MEM(RDX,RSI,1))
            VFMADD231PD(YMM( 9), YMM(1), MEM(RDX,RSI,2))
            VFMADD231PD(YMM(11), YMM(1), MEM(RDX,R13,1))
            VMOVUPD(MEM(RDX      ), YMM( 5))
            VMOVUPD(MEM(RDX,RSI,1), YMM( 7))
            VMOVUPD(MEM(RDX,RSI,2), YMM( 9))
            VMOVUPD(MEM(RDX,R13,1), YMM(11))

            VUNPCKLPD(YMM(2), YMM(13), YMM(15))
            VUNPCKHPD(YMM(3), YMM(13), YMM(15))
            VMOVUPD(XMM(13), MEM(RDX,      32))
            VMOVUPD(XMM(15), MEM(RDX,RSI,1,32))
            VINSERTF128(YMM(13), YMM(13), MEM(RDX,RSI,2,32), IMM(1))
            VINSERTF128(YMM(15), YMM(15), MEM(RDX,R13,1,32), IMM(1))
            VFMADD231PD(YMM(2), YMM(1), YMM(13))
            VFMADD231PD(YMM(3), YMM(1), YMM(15))
            VMOVUPD(MEM(RDX,      32), XMM(2))
            VMOVUPD(MEM(RDX,RSI,1,32), XMM(3))
            VEXTRACTF128(MEM(RDX,RSI,2,32), YMM(2), IMM(1))
            VEXTRACTF128(MEM(RDX,R13,1,32), YMM(3), IMM(1))

    JMP(.DDONE)
    LABEL(.DBETAZERO)

        CMP(RSI, IMM(8))
        JE(.DROWSTORBZ)

#if AMBI_METHOD != GS
        CMP(RDI, IMM(8))
        JE(.DCOLSTORBZ)
#endif

        LABEL(.DGENSTORBZ)

            DGEMM_GS_BZ(4)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(6)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(8)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(10)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(12)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(14)
            MOV(RCX, RDX)
            DGEMM_GS_BZ(5)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(7)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(9)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(11)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(13)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(15)

        JMP(.DDONE)
        LABEL(.DROWSTORBZ)

            VMOVUPD(MEM(RCX), YMM(4))
            VMOVUPD(MEM(RDX), YMM(5))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VMOVUPD(MEM(RCX), YMM(6))
            VMOVUPD(MEM(RDX), YMM(7))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VMOVUPD(MEM(RCX), YMM(8))
            VMOVUPD(MEM(RDX), YMM(9))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VMOVUPD(MEM(RCX), YMM(10))
            VMOVUPD(MEM(RDX), YMM(11))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VMOVUPD(MEM(RCX), YMM(12))
            VMOVUPD(MEM(RDX), YMM(13))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VMOVUPD(MEM(RCX), YMM(14))
            VMOVUPD(MEM(RDX), YMM(15))

        JMP(.DDONE)
        LABEL(.DCOLSTORBZ)

            //
            // Transpose and write out in four quadrants:
            //
            // +---------------+---------------+
            // |c00-c01-c02-c03|c04-c05-c06-c07|
            // |               |               |      +------+------+
            // |c10-c11-c12-c13|c14-c15-c16-c17|      |      |      |
            // |               |               |      |  Q1  |  Q3  |
            // |c20-c21-c22-c23|c24-c25-c26-c27|      |      |      |
            // |               |               |   =  +------+------+
            // |c30-c31-c32-c33|c34-c35-c36-c37|      |      |      |
            // +---------------+---------------+      |  Q2  |  Q4  |
            // |c40-c41-c42-c43|c44-c45-c46-c47|      |      |      |
            // |               |               |      +------+------+
            // |c50-c51-c52-c53|c54-c55-c56-c57|
            // +---------------+---------------+
            //
            //                ||
            //                \/
            //
            // +---------------+---------------+
            // |c00 c01 c02 c03|c04 c05 c06 c07|
            // | |   |   |   | | |   |   |   | |
            // |c10 c11 c12 c13|c14 c15 c16 c17|
            // | |   |   |   | | |   |   |   | |
            // |c20 c21 c22 c23|c24 c25 c26 c27|
            // | |   |   |   | | |   |   |   | |
            // |c30 c31 c32 c33|c34 c35 c36 c37|
            // +---------------+---------------+
            // |c40 c41 c42 c43|c44 c45 c46 c47|
            // | |   |   |   | | |   |   |   | |
            // |c50 c51 c52 c53|c54 c55 c56 c57|
            // +---------------+---------------+
            //

            VUNPCKLPD(YMM(2), YMM(4), YMM( 6))
            VUNPCKHPD(YMM(6), YMM(4), YMM( 6))
            VUNPCKHPD(YMM(3), YMM(8), YMM(10))
            VUNPCKLPD(YMM(8), YMM(8), YMM(10))
            VPERM2F128(YMM( 4), YMM(2), YMM(8), IMM(0x21))
            VPERM2F128(YMM(10), YMM(6), YMM(3), IMM(0x21))
            VBLENDPD(YMM( 8), YMM( 8), YMM( 4), IMM(0x3))
            VBLENDPD(YMM( 4), YMM( 4), YMM( 2), IMM(0x3))
            VBLENDPD(YMM( 6), YMM(10), YMM( 6), IMM(0x3))
            VBLENDPD(YMM(10), YMM( 3), YMM(10), IMM(0x3))
            VMOVUPD(MEM(RCX      ), YMM( 4))
            VMOVUPD(MEM(RCX,RSI,1), YMM( 6))
            VMOVUPD(MEM(RCX,RSI,2), YMM( 8))
            VMOVUPD(MEM(RCX,R13,1), YMM(10))

            VUNPCKLPD(YMM(2), YMM(12), YMM(14))
            VUNPCKHPD(YMM(3), YMM(12), YMM(14))
            VMOVUPD(MEM(RCX,      32), XMM(2))
            VMOVUPD(MEM(RCX,RSI,1,32), XMM(3))
            VEXTRACTF128(MEM(RCX,RSI,2,32), YMM(2), IMM(1))
            VEXTRACTF128(MEM(RCX,R13,1,32), YMM(3), IMM(1))

            VUNPCKLPD(YMM(2), YMM(5), YMM( 7))
            VUNPCKHPD(YMM(7), YMM(5), YMM( 7))
            VUNPCKHPD(YMM(3), YMM(9), YMM(11))
            VUNPCKLPD(YMM(9), YMM(9), YMM(11))
            VPERM2F128(YMM( 5), YMM(2), YMM(9), IMM(0x21))
            VPERM2F128(YMM(11), YMM(7), YMM(3), IMM(0x21))
            VBLENDPD(YMM( 9), YMM( 9), YMM( 5), IMM(0x3))
            VBLENDPD(YMM( 5), YMM( 5), YMM( 2), IMM(0x3))
            VBLENDPD(YMM( 7), YMM(11), YMM( 7), IMM(0x3))
            VBLENDPD(YMM(11), YMM( 3), YMM(11), IMM(0x3))
            VMOVUPD(MEM(RDX      ), YMM( 5))
            VMOVUPD(MEM(RDX,RSI,1), YMM( 7))
            VMOVUPD(MEM(RDX,RSI,2), YMM( 9))
            VMOVUPD(MEM(RDX,R13,1), YMM(11))

            VUNPCKLPD(YMM(2), YMM(13), YMM(15))
            VUNPCKHPD(YMM(3), YMM(13), YMM(15))
            VMOVUPD(MEM(RDX,      32), XMM(2))
            VMOVUPD(MEM(RDX,RSI,1,32), XMM(3))
            VEXTRACTF128(MEM(RDX,RSI,2,32), YMM(2), IMM(1))
            VEXTRACTF128(MEM(RDX,R13,1,32), YMM(3), IMM(1))

    LABEL(.DDONE)

    VZEROUPPER()

    END_ASM
    (
        : // output operands (none)
        : // input operands
          [k]     "m" (k),
          [a]     "m" (a),
          [b]     "m" (b),
          [alpha] "m" (alpha),
          [beta]  "m" (beta),
          [c]     "m" (c),
          [rs_c]  "m" (rs_c),
          [cs_c]  "m" (cs_c)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
          "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
          "xmm0", "xmm1", "xmm2", "xmm3",
          "xmm4", "xmm5", "xmm6", "xmm7",
          "xmm8", "xmm9", "xmm10", "xmm11",
          "xmm12", "xmm13", "xmm14", "xmm15",
          "memory"
    )
}

void bli_dgemm_asm_8x6
     (
       dim_t               k_,
       double*    restrict alpha,
       double*    restrict b,
       double*    restrict a,
       double*    restrict beta,
       double*    restrict c, inc_t cs_c_, inc_t rs_c_,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    //void*   a_next = bli_auxinfo_next_a( data );
    //void*   b_next = bli_auxinfo_next_b( data );

    int64_t k = k_;
    int64_t rs_c = rs_c_;
    int64_t cs_c = cs_c_;

    BEGIN_ASM

    VZEROALL()

    MOV(RAX, VAR(a))
    MOV(RBX, VAR(b))

    VMOVAPS(YMM(0), MEM(RBX,0*32)) // initialize loop by pre-loading
    VMOVAPS(YMM(1), MEM(RBX,1*32))

    ADD(RBX, IMM(32*6))

    MOV(RCX, VAR(c))
    MOV(RDI, VAR(rs_c))
    MOV(RSI, VAR(cs_c))
    LEA(RDI, MEM(,RDI,8)) // rs_c *= sizeof(double)
    LEA(RSI, MEM(,RSI,8)) // cs_c *= sizeof(double)

    CMP(RSI, IMM(8))
    JNE(.DROWSTORPF)

        LEA(R13, MEM(RDI,RDI,2)) // r13 = 3*rs_c
        LEA(R14, MEM(RDI,RDI,4)) // r14 = 5*rs_c

#if PF_C_L2 == PF_IR || PF_C_L2 == PF_NEAR
        PREFETCH_C_6(1,RCX,RDI,R13,R14,64)
#else
        LEA(RDX, MEM(RCX,R13,2)) // rdx = c + 6*rs_c;
        PREFETCH_C_6(1,RDX,RDI,R13,R14,0)
#endif

        PREFETCH_C_6(0,RCX,RDI,R13,R14,0)

    JMP(.DACCUM)
    LABEL(.DROWSTORPF)

        LEA(R13, MEM(RSI,RSI,2)) // r13 = 3*cs_c
        LEA(R14, MEM(RSI,RSI,4)) // r14 = 5*cs_c
        LEA(R15, MEM(RSI,R13,2)) // r15 = 7*cs_c

#if PF_C_L2 == PF_JR || PF_C_L2 == PF_NEAR
        PREFETCH_C_8(1,RCX,RSI,R13,R14,R15,48)
#else
        LEA(RDX, MEM(RCX,RSI,8)) // rdx = c + 8*cs_c;
        PREFETCH_C_8(1,RDX,RSI,R13,R14,R15,0)
#endif

        PREFETCH_C_8(0,RCX,RSI,R13,R14,R15,0)

    LABEL(.DACCUM)

    MOV(RSI, VAR(k))
    MOV(R8, RSI)
    SAR(RSI, IMM(2))
    JZ(.DCONSIDKLEFT)

    LABEL(.DLOOPKITER)

        PREFETCH(0, MEM(RBX,64*8))

        DGEMM_ITERATION(0)

        PREFETCH(0, MEM(RBX,72*8))

        DGEMM_ITERATION(1)

        PREFETCH(0, MEM(RBX,80*8))

        DGEMM_ITERATION(2)

        PREFETCH(0, MEM(RBX,88*8))

        DGEMM_ITERATION(3)

        ADD(RAX, IMM(4*6*8))
        ADD(RBX, IMM(4*8*8))

        DEC(RSI)

    JNZ(.DLOOPKITER)

    LABEL(.DCONSIDKLEFT)

    MOV(RSI, R8)
    AND(RSI, IMM(3))
    JZ(.DPOSTACCUM)

    LABEL(.DLOOPKLEFT)

        PREFETCH(0, MEM(RBX,64*8))

        DGEMM_ITERATION(0)

        ADD(RAX, IMM(1*6*8))
        ADD(RBX, IMM(1*8*8))

        DEC(RSI)

    JNZ(.DLOOPKLEFT)

    LABEL(.DPOSTACCUM)

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSD(YMM(0), MEM(RAX))
    VBROADCASTSD(YMM(1), MEM(RBX))

    VMULPD(YMM( 4), YMM( 4), YMM(0))
    VMULPD(YMM( 5), YMM( 5), YMM(0))
    VMULPD(YMM( 6), YMM( 6), YMM(0))
    VMULPD(YMM( 7), YMM( 7), YMM(0))
    VMULPD(YMM( 8), YMM( 8), YMM(0))
    VMULPD(YMM( 9), YMM( 9), YMM(0))
    VMULPD(YMM(10), YMM(10), YMM(0))
    VMULPD(YMM(11), YMM(11), YMM(0))
    VMULPD(YMM(12), YMM(12), YMM(0))
    VMULPD(YMM(13), YMM(13), YMM(0))
    VMULPD(YMM(14), YMM(14), YMM(0))
    VMULPD(YMM(15), YMM(15), YMM(0))

    MOV(RSI, VAR(cs_c))
    LEA(RSI, MEM(,RSI,8))

    LEA(RDX, MEM(RCX,RSI,4))

    LEA(R13, MEM(RSI,RSI,2))

    VXORPD(YMM(0), YMM(0), YMM(0))
    VUCOMISD(XMM(1), XMM(0))
    JZ(.DBETAZERO)

        CMP(RSI, IMM(8))
        JE(.DCOLSTORED)

#if AMBI_METHOD != GS
        CMP(RDI, IMM(8))
        JE(.DROWSTORED)
#endif

        LABEL(.DGENSTORED)

            DGEMM_GS(4)
            ADD(RCX, RDI)
            DGEMM_GS(6)
            ADD(RCX, RDI)
            DGEMM_GS(8)
            ADD(RCX, RDI)
            DGEMM_GS(10)
            ADD(RCX, RDI)
            DGEMM_GS(12)
            ADD(RCX, RDI)
            DGEMM_GS(14)
            MOV(RCX, RDX)
            DGEMM_GS(5)
            ADD(RCX, RDI)
            DGEMM_GS(7)
            ADD(RCX, RDI)
            DGEMM_GS(9)
            ADD(RCX, RDI)
            DGEMM_GS(11)
            ADD(RCX, RDI)
            DGEMM_GS(13)
            ADD(RCX, RDI)
            DGEMM_GS(15)

        JMP(.DDONE)
        LABEL(.DCOLSTORED)

            VFMADD231PD(YMM(4), YMM(1), MEM(RCX))
            VFMADD231PD(YMM(5), YMM(1), MEM(RDX))
            VMOVUPD(MEM(RCX), YMM(4))
            VMOVUPD(MEM(RDX), YMM(5))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VFMADD231PD(YMM(6), YMM(1), MEM(RCX))
            VFMADD231PD(YMM(7), YMM(1), MEM(RDX))
            VMOVUPD(MEM(RCX), YMM(6))
            VMOVUPD(MEM(RDX), YMM(7))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VFMADD231PD(YMM(8), YMM(1), MEM(RCX))
            VFMADD231PD(YMM(9), YMM(1), MEM(RDX))
            VMOVUPD(MEM(RCX), YMM(8))
            VMOVUPD(MEM(RDX), YMM(9))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VFMADD231PD(YMM(10), YMM(1), MEM(RCX))
            VFMADD231PD(YMM(11), YMM(1), MEM(RDX))
            VMOVUPD(MEM(RCX), YMM(10))
            VMOVUPD(MEM(RDX), YMM(11))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VFMADD231PD(YMM(12), YMM(1), MEM(RCX))
            VFMADD231PD(YMM(13), YMM(1), MEM(RDX))
            VMOVUPD(MEM(RCX), YMM(12))
            VMOVUPD(MEM(RDX), YMM(13))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VFMADD231PD(YMM(14), YMM(1), MEM(RCX))
            VFMADD231PD(YMM(15), YMM(1), MEM(RDX))
            VMOVUPD(MEM(RCX), YMM(14))
            VMOVUPD(MEM(RDX), YMM(15))

        JMP(.DDONE)
        LABEL(.DROWSTORED)

            //
            // Transpose and write out in four quadrants:
            //
            // +---------------+---------------+
            // |c00-c01-c02-c03|c04-c05-c06-c07|
            // |               |               |      +------+------+
            // |c10-c11-c12-c13|c14-c15-c16-c17|      |      |      |
            // |               |               |      |  Q1  |  Q3  |
            // |c20-c21-c22-c23|c24-c25-c26-c27|      |      |      |
            // |               |               |   =  +------+------+
            // |c30-c31-c32-c33|c34-c35-c36-c37|      |      |      |
            // +---------------+---------------+      |  Q2  |  Q4  |
            // |c40-c41-c42-c43|c44-c45-c46-c47|      |      |      |
            // |               |               |      +------+------+
            // |c50-c51-c52-c53|c54-c55-c56-c57|
            // +---------------+---------------+
            //
            //                ||
            //                \/
            //
            // +---------------+---------------+
            // |c00 c01 c02 c03|c04 c05 c06 c07|
            // | |   |   |   | | |   |   |   | |
            // |c10 c11 c12 c13|c14 c15 c16 c17|
            // | |   |   |   | | |   |   |   | |
            // |c20 c21 c22 c23|c24 c25 c26 c27|
            // | |   |   |   | | |   |   |   | |
            // |c30 c31 c32 c33|c34 c35 c36 c37|
            // +---------------+---------------+
            // |c40 c41 c42 c43|c44 c45 c46 c47|
            // | |   |   |   | | |   |   |   | |
            // |c50 c51 c52 c53|c54 c55 c56 c57|
            // +---------------+---------------+
            //

            VUNPCKLPD(YMM(2), YMM(4), YMM( 6))
            VUNPCKHPD(YMM(6), YMM(4), YMM( 6))
            VUNPCKHPD(YMM(3), YMM(8), YMM(10))
            VUNPCKLPD(YMM(8), YMM(8), YMM(10))
            VPERM2F128(YMM( 4), YMM(2), YMM(8), IMM(0x21))
            VPERM2F128(YMM(10), YMM(6), YMM(3), IMM(0x21))
            VBLENDPD(YMM( 8), YMM( 8), YMM( 4), IMM(0x3))
            VBLENDPD(YMM( 4), YMM( 4), YMM( 2), IMM(0x3))
            VBLENDPD(YMM( 6), YMM(10), YMM( 6), IMM(0x3))
            VBLENDPD(YMM(10), YMM( 3), YMM(10), IMM(0x3))
            VFMADD231PD(YMM( 4), YMM(1), MEM(RCX      ))
            VFMADD231PD(YMM( 6), YMM(1), MEM(RCX,RSI,1))
            VFMADD231PD(YMM( 8), YMM(1), MEM(RCX,RSI,2))
            VFMADD231PD(YMM(10), YMM(1), MEM(RCX,R13,1))
            VMOVUPD(MEM(RCX      ), YMM( 4))
            VMOVUPD(MEM(RCX,RSI,1), YMM( 6))
            VMOVUPD(MEM(RCX,RSI,2), YMM( 8))
            VMOVUPD(MEM(RCX,R13,1), YMM(10))

            VUNPCKLPD(YMM(2), YMM(12), YMM(14))
            VUNPCKHPD(YMM(3), YMM(12), YMM(14))
            VMOVUPD(XMM(12), MEM(RCX,      32))
            VMOVUPD(XMM(14), MEM(RCX,RSI,1,32))
            VINSERTF128(YMM(12), YMM(12), MEM(RCX,RSI,2,32), IMM(1))
            VINSERTF128(YMM(14), YMM(14), MEM(RCX,R13,1,32), IMM(1))
            VFMADD231PD(YMM(2), YMM(1), YMM(12))
            VFMADD231PD(YMM(3), YMM(1), YMM(14))
            VMOVUPD(MEM(RCX,      32), XMM(2))
            VMOVUPD(MEM(RCX,RSI,1,32), XMM(3))
            VEXTRACTF128(MEM(RCX,RSI,2,32), YMM(2), IMM(1))
            VEXTRACTF128(MEM(RCX,R13,1,32), YMM(3), IMM(1))

            VUNPCKLPD(YMM(2), YMM(5), YMM( 7))
            VUNPCKHPD(YMM(7), YMM(5), YMM( 7))
            VUNPCKHPD(YMM(3), YMM(9), YMM(11))
            VUNPCKLPD(YMM(9), YMM(9), YMM(11))
            VPERM2F128(YMM( 5), YMM(2), YMM(9), IMM(0x21))
            VPERM2F128(YMM(11), YMM(7), YMM(3), IMM(0x21))
            VBLENDPD(YMM( 9), YMM( 9), YMM( 5), IMM(0x3))
            VBLENDPD(YMM( 5), YMM( 5), YMM( 2), IMM(0x3))
            VBLENDPD(YMM( 7), YMM(11), YMM( 7), IMM(0x3))
            VBLENDPD(YMM(11), YMM( 3), YMM(11), IMM(0x3))
            VFMADD231PD(YMM( 5), YMM(1), MEM(RDX      ))
            VFMADD231PD(YMM( 7), YMM(1), MEM(RDX,RSI,1))
            VFMADD231PD(YMM( 9), YMM(1), MEM(RDX,RSI,2))
            VFMADD231PD(YMM(11), YMM(1), MEM(RDX,R13,1))
            VMOVUPD(MEM(RDX      ), YMM( 5))
            VMOVUPD(MEM(RDX,RSI,1), YMM( 7))
            VMOVUPD(MEM(RDX,RSI,2), YMM( 9))
            VMOVUPD(MEM(RDX,R13,1), YMM(11))

            VUNPCKLPD(YMM(2), YMM(13), YMM(15))
            VUNPCKHPD(YMM(3), YMM(13), YMM(15))
            VMOVUPD(XMM(13), MEM(RDX,      32))
            VMOVUPD(XMM(15), MEM(RDX,RSI,1,32))
            VINSERTF128(YMM(13), YMM(13), MEM(RDX,RSI,2,32), IMM(1))
            VINSERTF128(YMM(15), YMM(15), MEM(RDX,R13,1,32), IMM(1))
            VFMADD231PD(YMM(2), YMM(1), YMM(13))
            VFMADD231PD(YMM(3), YMM(1), YMM(15))
            VMOVUPD(MEM(RDX,      32), XMM(2))
            VMOVUPD(MEM(RDX,RSI,1,32), XMM(3))
            VEXTRACTF128(MEM(RDX,RSI,2,32), YMM(2), IMM(1))
            VEXTRACTF128(MEM(RDX,R13,1,32), YMM(3), IMM(1))

    JMP(.DDONE)
    LABEL(.DBETAZERO)

        CMP(RSI, IMM(8))
        JE(.DCOLSTORBZ)

#if AMBI_METHOD != GS
        CMP(RDI, IMM(8))
        JE(.DROWSTORBZ)
#endif

        LABEL(.DGENSTORBZ)

            DGEMM_GS_BZ(4)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(6)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(8)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(10)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(12)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(14)
            MOV(RCX, RDX)
            DGEMM_GS_BZ(5)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(7)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(9)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(11)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(13)
            ADD(RCX, RDI)
            DGEMM_GS_BZ(15)

        JMP(.DDONE)
        LABEL(.DCOLSTORBZ)

            VMOVUPD(MEM(RCX), YMM(4))
            VMOVUPD(MEM(RDX), YMM(5))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VMOVUPD(MEM(RCX), YMM(6))
            VMOVUPD(MEM(RDX), YMM(7))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VMOVUPD(MEM(RCX), YMM(8))
            VMOVUPD(MEM(RDX), YMM(9))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VMOVUPD(MEM(RCX), YMM(10))
            VMOVUPD(MEM(RDX), YMM(11))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VMOVUPD(MEM(RCX), YMM(12))
            VMOVUPD(MEM(RDX), YMM(13))
            ADD(RCX, RDI)
            ADD(RDX, RDI)

            VMOVUPD(MEM(RCX), YMM(14))
            VMOVUPD(MEM(RDX), YMM(15))

        JMP(.DDONE)
        LABEL(.DROWSTORBZ)

            //
            // Transpose and write out in four quadrants:
            //
            // +---------------+---------------+
            // |c00-c01-c02-c03|c04-c05-c06-c07|
            // |               |               |      +------+------+
            // |c10-c11-c12-c13|c14-c15-c16-c17|      |      |      |
            // |               |               |      |  Q1  |  Q3  |
            // |c20-c21-c22-c23|c24-c25-c26-c27|      |      |      |
            // |               |               |   =  +------+------+
            // |c30-c31-c32-c33|c34-c35-c36-c37|      |      |      |
            // +---------------+---------------+      |  Q2  |  Q4  |
            // |c40-c41-c42-c43|c44-c45-c46-c47|      |      |      |
            // |               |               |      +------+------+
            // |c50-c51-c52-c53|c54-c55-c56-c57|
            // +---------------+---------------+
            //
            //                ||
            //                \/
            //
            // +---------------+---------------+
            // |c00 c01 c02 c03|c04 c05 c06 c07|
            // | |   |   |   | | |   |   |   | |
            // |c10 c11 c12 c13|c14 c15 c16 c17|
            // | |   |   |   | | |   |   |   | |
            // |c20 c21 c22 c23|c24 c25 c26 c27|
            // | |   |   |   | | |   |   |   | |
            // |c30 c31 c32 c33|c34 c35 c36 c37|
            // +---------------+---------------+
            // |c40 c41 c42 c43|c44 c45 c46 c47|
            // | |   |   |   | | |   |   |   | |
            // |c50 c51 c52 c53|c54 c55 c56 c57|
            // +---------------+---------------+
            //

            VUNPCKLPD(YMM(2), YMM(4), YMM( 6))
            VUNPCKHPD(YMM(6), YMM(4), YMM( 6))
            VUNPCKHPD(YMM(3), YMM(8), YMM(10))
            VUNPCKLPD(YMM(8), YMM(8), YMM(10))
            VPERM2F128(YMM( 4), YMM(2), YMM(8), IMM(0x21))
            VPERM2F128(YMM(10), YMM(6), YMM(3), IMM(0x21))
            VBLENDPD(YMM( 8), YMM( 8), YMM( 4), IMM(0x3))
            VBLENDPD(YMM( 4), YMM( 4), YMM( 2), IMM(0x3))
            VBLENDPD(YMM( 6), YMM(10), YMM( 6), IMM(0x3))
            VBLENDPD(YMM(10), YMM( 3), YMM(10), IMM(0x3))
            VMOVUPD(MEM(RCX      ), YMM( 4))
            VMOVUPD(MEM(RCX,RSI,1), YMM( 6))
            VMOVUPD(MEM(RCX,RSI,2), YMM( 8))
            VMOVUPD(MEM(RCX,R13,1), YMM(10))

            VUNPCKLPD(YMM(2), YMM(12), YMM(14))
            VUNPCKHPD(YMM(3), YMM(12), YMM(14))
            VMOVUPD(MEM(RCX,      32), XMM(2))
            VMOVUPD(MEM(RCX,RSI,1,32), XMM(3))
            VEXTRACTF128(MEM(RCX,RSI,2,32), YMM(2), IMM(1))
            VEXTRACTF128(MEM(RCX,R13,1,32), YMM(3), IMM(1))

            VUNPCKLPD(YMM(2), YMM(5), YMM( 7))
            VUNPCKHPD(YMM(7), YMM(5), YMM( 7))
            VUNPCKHPD(YMM(3), YMM(9), YMM(11))
            VUNPCKLPD(YMM(9), YMM(9), YMM(11))
            VPERM2F128(YMM( 5), YMM(2), YMM(9), IMM(0x21))
            VPERM2F128(YMM(11), YMM(7), YMM(3), IMM(0x21))
            VBLENDPD(YMM( 9), YMM( 9), YMM( 5), IMM(0x3))
            VBLENDPD(YMM( 5), YMM( 5), YMM( 2), IMM(0x3))
            VBLENDPD(YMM( 7), YMM(11), YMM( 7), IMM(0x3))
            VBLENDPD(YMM(11), YMM( 3), YMM(11), IMM(0x3))
            VMOVUPD(MEM(RDX      ), YMM( 5))
            VMOVUPD(MEM(RDX,RSI,1), YMM( 7))
            VMOVUPD(MEM(RDX,RSI,2), YMM( 9))
            VMOVUPD(MEM(RDX,R13,1), YMM(11))

            VUNPCKLPD(YMM(2), YMM(13), YMM(15))
            VUNPCKHPD(YMM(3), YMM(13), YMM(15))
            VMOVUPD(MEM(RDX,      32), XMM(2))
            VMOVUPD(MEM(RDX,RSI,1,32), XMM(3))
            VEXTRACTF128(MEM(RDX,RSI,2,32), YMM(2), IMM(1))
            VEXTRACTF128(MEM(RDX,R13,1,32), YMM(3), IMM(1))

    LABEL(.DDONE)

    VZEROUPPER()

    END_ASM
    (
        : // output operands (none)
        : // input operands
          [k]     "m" (k),
          [a]     "m" (a),
          [b]     "m" (b),
          [alpha] "m" (alpha),
          [beta]  "m" (beta),
          [c]     "m" (c),
          [rs_c]  "m" (rs_c),
          [cs_c]  "m" (cs_c)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
          "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
          "xmm0", "xmm1", "xmm2", "xmm3",
          "xmm4", "xmm5", "xmm6", "xmm7",
          "xmm8", "xmm9", "xmm10", "xmm11",
          "xmm12", "xmm13", "xmm14", "xmm15",
          "memory"
    )
}

#if 0

void bli_cgemm_asm_
     (
       dim_t               k,
       scomplex*  restrict alpha,
       scomplex*  restrict a,
       scomplex*  restrict b,
       scomplex*  restrict beta,
       scomplex*  restrict c, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	//dim_t   k_iter = k / 4;
	//dim_t   k_left = k % 4;

}



void bli_zgemm_asm_
     (
       dim_t               k,
       dcomplex*  restrict alpha,
       dcomplex*  restrict a,
       dcomplex*  restrict b,
       dcomplex*  restrict beta,
       dcomplex*  restrict c, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	//dim_t   k_iter = k / 4;
	//dim_t   k_left = k % 4;

}

#endif
