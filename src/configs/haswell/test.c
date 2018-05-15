#include <string.h>
#include <stdio.h>
#include "util/asm_x86.h"

int main()
{
    double betas[2] = {0.0, -1.0};

    for (int ibeta = 0;ibeta < 2;ibeta++)
    {

    float salpha_ = 1.0;
    float sbeta_ = betas[ibeta];
    float* salpha = &salpha_;
    float* sbeta = &sbeta_;

    float s1[6][16];
    float s2[16][6];

    for (int i = 0;i < 6;i++)
        for (int j = 0;j < 16;j++)
            s1[i][j] = i*16+j;

    for (int i = 0;i < 6;i++)
        for (int j = 0;j < 16;j++)
            s2[j][i] = -s1[i][j]*sbeta_;

    float* sa = &s1[0][0];
    float* sb = &s2[0][0];

	BEGIN_ASM

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSS(YMM(0), MEM(RAX))
    VBROADCASTSS(YMM(1), MEM(RBX))

    MOV(RSI, VAR(a))
    MOV(RCX, VAR(b))

    VMOVUPS(YMM( 4), MEM(RSI, 0*32))
    VMOVUPS(YMM( 5), MEM(RSI, 1*32))
    VMOVUPS(YMM( 6), MEM(RSI, 2*32))
    VMOVUPS(YMM( 7), MEM(RSI, 3*32))
    VMOVUPS(YMM( 8), MEM(RSI, 4*32))
    VMOVUPS(YMM( 9), MEM(RSI, 5*32))
    VMOVUPS(YMM(10), MEM(RSI, 6*32))
    VMOVUPS(YMM(11), MEM(RSI, 7*32))
    VMOVUPS(YMM(12), MEM(RSI, 8*32))
    VMOVUPS(YMM(13), MEM(RSI, 9*32))
    VMOVUPS(YMM(14), MEM(RSI,10*32))
    VMOVUPS(YMM(15), MEM(RSI,11*32))

    //MOV(RSI, VAR(cs_c))
    MOV(RSI, IMM(6))
    LEA(RSI, MEM(,RSI,4))

    LEA(RDX, MEM(RCX,RSI,8))

    LEA(R13, MEM(RSI,RSI,2))
    LEA(R15, MEM(RSI,RSI,4))
    LEA(R10, MEM(R13,RSI,4))

    VXORPS(YMM(0), YMM(0), YMM(0))
    VUCOMISS(XMM(1), XMM(0))
    JZ(.SBETAZERO)

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
          [a]     "m" (sa),
          [b]     "m" (sb),
          [alpha] "m" (salpha),
          [beta]  "m" (sbeta)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
          "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
          "xmm0", "xmm1", "xmm2", "xmm3",
          "xmm4", "xmm5", "xmm6", "xmm7",
          "xmm8", "xmm9", "xmm10", "xmm11",
          "xmm12", "xmm13", "xmm14", "xmm15",
          "memory"
	)

    double dalpha_ = 1.0;
	double dbeta_ = betas[ibeta];
	double* dalpha = &dalpha_;
	double* dbeta = &dbeta_;

	double d1[6][8];
	double d2[8][6];

    for (int i = 0;i < 6;i++)
        for (int j = 0;j < 8;j++)
            d1[i][j] = i*16+j;

    for (int i = 0;i < 6;i++)
        for (int j = 0;j < 8;j++)
            d2[j][i] = -d1[i][j]*dbeta_;

    double* da = &d1[0][0];
    double* db = &d2[0][0];

    BEGIN_ASM

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSD(YMM(0), MEM(RAX))
    VBROADCASTSD(YMM(1), MEM(RBX))

    MOV(RSI, VAR(a))
    MOV(RCX, VAR(b))

    VMOVUPD(YMM( 4), MEM(RSI, 0*32))
    VMOVUPD(YMM( 5), MEM(RSI, 1*32))
    VMOVUPD(YMM( 6), MEM(RSI, 2*32))
    VMOVUPD(YMM( 7), MEM(RSI, 3*32))
    VMOVUPD(YMM( 8), MEM(RSI, 4*32))
    VMOVUPD(YMM( 9), MEM(RSI, 5*32))
    VMOVUPD(YMM(10), MEM(RSI, 6*32))
    VMOVUPD(YMM(11), MEM(RSI, 7*32))
    VMOVUPD(YMM(12), MEM(RSI, 8*32))
    VMOVUPD(YMM(13), MEM(RSI, 9*32))
    VMOVUPD(YMM(14), MEM(RSI,10*32))
    VMOVUPD(YMM(15), MEM(RSI,11*32))

    //MOV(RSI, VAR(cs_c))
    MOV(RSI, IMM(6))
    LEA(RSI, MEM(,RSI,8))

    LEA(RDX, MEM(RCX,RSI,4))

    LEA(R13, MEM(RSI,RSI,2))

    VXORPD(YMM(0), YMM(0), YMM(0))
    VUCOMISD(XMM(1), XMM(0))
    JZ(.DBETAZERO)

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
          [a]     "m" (da),
          [b]     "m" (db),
          [alpha] "m" (dalpha),
          [beta]  "m" (dbeta)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
          "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
          "xmm0", "xmm1", "xmm2", "xmm3",
          "xmm4", "xmm5", "xmm6", "xmm7",
          "xmm8", "xmm9", "xmm10", "xmm11",
          "xmm12", "xmm13", "xmm14", "xmm15",
          "memory"
    )

    printf("float %f:\n\n", *sbeta);

    for (int i = 0;i < 6;i++)
    {
        for (int j = 0;j < 16;j++)
            printf("%02x ", (int)s1[i][j]);
        printf("\n");
    }
    printf("\n");

    for (int i = 0;i < 6;i++)
    {
        for (int j = 0;j < 16;j++)
            printf("%02x ", (int)s2[j][i]);
        printf("\n");
    }
    printf("\n");

    printf("double %f:\n\n", *dbeta);

    for (int i = 0;i < 6;i++)
    {
        for (int j = 0;j < 8;j++)
            printf("%02x ", (int)d1[i][j]);
        printf("\n");
    }
    printf("\n");

    for (int i = 0;i < 6;i++)
    {
        for (int j = 0;j < 8;j++)
            printf("%02x ", (int)d2[j][i]);
        printf("\n");
    }
    printf("\n");

    }
}
