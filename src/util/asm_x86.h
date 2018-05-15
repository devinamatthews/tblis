#ifndef _TBLIS_UTIL_ASM_X86_H_
#define _TBLIS_UTIL_ASM_X86_H_

//
// Assembly macros to make inline x86 with AT&T syntax somewhat less painful
//

#define STRINGIFY(...) #__VA_ARGS__
#define GET_MACRO(_1,_2,_3,_4,NAME,...) NAME

#if defined(_WIN32) || defined(__MIC__)

// Intel-style assembly blocks

#define BEGIN_ASM __asm {

#define END_ASM(...) }

#define ASM(...) __asm __VA_ARGS__
#define LABEL(label) label:
#define REGISTER(r) r
#define IMM(x) x
#define VAR(x) x
#define MASK(x) { x }

#define MEM_4(reg,off,scale,disp) [reg + off*scale + disp]
#define MEM_3(reg,off,scale) [reg + off*scale]
#define MEM_2(reg,disp) [reg + disp]
#define MEM_1(reg) [reg]

#define INSTR_4(name,_0,_1,_2,_3) ASM(name _0,_1,_2,_3)
#define INSTR_3(name,_0,_1,_2) ASM(name _0,_1,_2)
#define INSTR_2(name,_0,_1) ASM(name _0,_1)
#define INSTR_1(name,_0) ASM(name _0)
#define INSTR_0(name) ASM(name)

#define ALIGN16 ASM(align 16)
#define ALIGN32 ASM(align 32)

#else

// GCC extended assembly with AT&T syntax

#define COMMENT_BEGIN "#"
#define COMMENT_END

#define BEGIN_ASM __asm__ volatile (

#define END_ASM(...) __VA_ARGS__ );

#define ASM(...) STRINGIFY(__VA_ARGS__) "\n\t"
#define LABEL(label) STRINGIFY(label) ":\n\t"
#define REGISTER(r) %%r
#define IMM(x) $##x
#define VAR(x) %[x]
#define MASK(x) %{x%}

#define MEM_4(reg,off,scale,disp) disp(reg,off,scale)
#define MEM_3(reg,off,scale) (reg,off,scale)
#define MEM_2(reg,disp) disp(reg)
#define MEM_1(reg) (reg)

#define INSTR_4(name,_0,_1,_2,_3) ASM(name _3,_2,_1,_0)
#define INSTR_3(name,_0,_1,_2) ASM(name _2,_1,_0)
#define INSTR_2(name,_0,_1) ASM(name _1,_0)
#define INSTR_1(name,_0) ASM(name _0)
#define INSTR_0(name) ASM(name)

#define ALIGN16 ASM(.p2align 4)
#define ALIGN32 ASM(.p2align 5)

#endif

// General-purpose registers

#define AL REGISTER(al)
#define AH REGISTER(ah)
#define BL REGISTER(bl)
#define BH REGISTER(bh)
#define CL REGISTER(cl)
#define CH REGISTER(ch)
#define DL REGISTER(dl)
#define DH REGISTER(dh)

#define EAX REGISTER(eax)
#define EBX REGISTER(ebx)
#define ECX REGISTER(ecx)
#define EDX REGISTER(edx)
#define EBP REGISTER(ebp)
#define EDI REGISTER(edi)
#define ESI REGISTER(esi)
#define R8D REGISTER(r8d)
#define R9D REGISTER(r9d)
#define R10D REGISTER(r10d)
#define R11D REGISTER(r11d)
#define R12D REGISTER(r12d)
#define R13D REGISTER(r13d)
#define R14D REGISTER(r14d)
#define R15D REGISTER(r15d)

#define RAX REGISTER(rax)
#define RBX REGISTER(rbx)
#define RCX REGISTER(rcx)
#define RDX REGISTER(rdx)
#define RBP REGISTER(rbp)
#define RDI REGISTER(rdi)
#define RSI REGISTER(rsi)
#define R8 REGISTER(r8)
#define R9 REGISTER(r9)
#define R10 REGISTER(r10)
#define R11 REGISTER(r11)
#define R12 REGISTER(r12)
#define R13 REGISTER(r13)
#define R14 REGISTER(r14)
#define R15 REGISTER(r15)

// Vector registers

#define XMM(x) REGISTER(xmm##x)
#define YMM(x) REGISTER(ymm##x)
#define ZMM(x) REGISTER(zmm##x)
#define K(x) REGISTER(k##x)
#define MASK_K(n) MASK(K(n))
#define MASK_KZ(n) MASK(K(n))MASK(z)

// Memory access

#define MEM(...) GET_MACRO(__VA_ARGS__,MEM_4,MEM_3,MEM_2,MEM_1)(__VA_ARGS__)
#define MEM_1TO8(...) MEM(__VA_ARGS__) MASK(1to8)
#define MEM_1TO16(...) MEM(__VA_ARGS__) MASK(1to16)
#define MEM_BCAST(...) MEM(__VA_ARGS__) MASK(b)

// Instructions

#define INSTR(name,...) GET_MACRO(__VA_ARGS__,INSTR_4,INSTR_3,INSTR_2, \
                                  INSTR_1,INSTR_0)(name,__VA_ARGS__)

// Jumps

#define JC(_0) INSTR(jc, _0)
#define JB JC
#define JNAE JC
#define JNC(_0) INSTR(jnc, _0)
#define JNB JNC
#define JAE JNC

#define JO(_0) INSTR(jo, _0)
#define JNO(_0) INSTR(jno, _0)

#define JP(_0) INSTR(jp, _0)
#define JPE JP
#define JNP(_0) INSTR(jnp, _0)
#define JPO JNP

#define JS(_0) INSTR(js, _0)
#define JNS(_0) INSTR(jns, _0)

#define JA(_0) INSTR(ja, _0)
#define JNBE JA
#define JNA(_0) INSTR(jna, _0)
#define JBE JNA

#define JL(_0) INSTR(jl, _0)
#define JNGE JL
#define JNL(_0) INSTR(jnl, _0)
#define JGE JNL

#define JG(_0) INSTR(jg, _0)
#define JNLE JG
#define JNG(_0) INSTR(jng, _0)
#define JLE JNG

#define JE(_0) INSTR(je, _0)
#define JZ JE
#define JNE(_0) INSTR(jne, _0)
#define JNZ JNE

#define JMP(_0) INSTR(jmp, _0)

#define SETE(_0) INSTR(sete, _0)
#define SETZ SETE

// Comparisons

#define CMP(_0, _1) INSTR(cmp, _0, _1)
#define TEST(_0, _1) INSTR(test, _0, _1)

// Integer math

#define AND(_0, _1) INSTR(and, _0, _1)
#define OR(_0, _1) INSTR(or, _0, _1)
#define XOR(_0, _1) INSTR(xor, _0, _1)
#define ADD(_0, _1) INSTR(add, _0, _1)
#define SUB(_0, _1) INSTR(sub, _0, _1)
#define SAL(...) INSTR(sal, __VA_ARGS__)
#define SAR(...) INSTR(sar, __VA_ARGS__)
#define SHLX(_0, _1, _2) INSTR(shlx, _0, _1, _2)
#define SHRX(_0, _1, _2) INSTR(shrx, _0, _1, _2)
#define DEC(_0) INSTR(dec _0)
#define INC(_0) INSTR(inc _0)

// Memory access

#define LEA(_0, _1) INSTR(lea, _0, _1)
#define MOV(_0, _1) INSTR(mov, _0, _1)
#define MOVD(_0, _1) INSTR(movd, _0, _1)
#define MOVL(_0, _1) INSTR(movl, _0, _1)
#define MOVQ(_0, _1) INSTR(movq, _0, _1)

// Vector moves

#define MOVSS(_0, _1) INSTR(movss, _0, _1)
#define MOVSD(_0, _1) INSTR(movsd, _0, _1)
#define MOVAPS(_0, _1) INSTR(movaps, _0, _1)
#define MOVAPD(_0, _1) INSTR(movaps, _0, _1) //use movaps because it is shorter
#define MOVDDUP(_0, _1) INSTR(movddup, _0, _1)
#define MOVLPS(_0, _1) INSTR(movlps, _0, _1)
#define MOVHPS(_0, _1) INSTR(movhps, _0, _1)
#define MOVLPD(_0, _1) INSTR(movlpd, _0, _1)
#define MOVHPD(_0, _1) INSTR(movhpd, _0, _1)

#define VMOVSLDUP(_0, _1) INSTR(vmovsldup, _0, _1)
#define VMOVSHDUP(_0, _1) INSTR(vmovshdup, _0, _1)
#define VMOVD(_0, _1) INSTR(vmovd, _0, _1)
#define VMOVQ(_0, _1) INSTR(vmovq, _0, _1)
#define VMOVSS(_0, _1) INSTR(vmovss, _0, _1)
#define VMOVSD(_0, _1) INSTR(vmovsd, _0, _1)
#define VMOVAPS(_0, _1) INSTR(vmovaps, _0, _1)
#define VMOVUPS(_0, _1) INSTR(vmovups, _0, _1)
#define VMOVAPD(_0, _1) INSTR(vmovapd, _0, _1)
#define VMOVUPD(_0, _1) INSTR(vmovupd, _0, _1)
#define VMOVLPS(...) INSTR(vmovlps, __VA_ARGS__)
#define VMOVHPS(...) INSTR(vmovhps, __VA_ARGS__)
#define VMOVLPD(...) INSTR(vmovlpd, __VA_ARGS__)
#define VMOVHPD(...) INSTR(vmovhpd, __VA_ARGS__)
#define VMOVDQA(_0, _1) INSTR(vmovdqa, _0, _1)
#define VMOVDQA32(_0, _1) INSTR(vmovdqa32, _0, _1)
#define VMOVDQA64(_0, _1) INSTR(vmovdqa64, _0, _1)
#define VBROADCASTSS(_0, _1) INSTR(vbroadcastss, _0, _1)
#define VBROADCASTSD(_0, _1) INSTR(vbroadcastsd, _0, _1)
#define VPBROADCASTD(_0, _1) INSTR(vpbroadcastd, _0, _1)
#define VPBROADCASTQ(_0, _1) INSTR(vpbroadcastq, _0, _1)
#define VBROADCASTF128(_0, _1) INSTR(vbroadcastf128, _0, _1)
#define VBROADCASTF64X4(_0, _1) INSTR(vbroadcastf64x4, _0, _1)
#define VGATHERDPS(_0, _1) INSTR(vgatherdps, _0, _1)
#define VSCATTERDPS(_0, _1) INSTR(vscatterdps, _0, _1)
#define VGATHERDPD(_0, _1) INSTR(vgatherdpd, _0, _1)
#define VSCATTERDPD(_0, _1) INSTR(vscatterdpd, _0, _1)
#define VGATHERQPS(_0, _1) INSTR(vgatherqps, _0, _1)
#define VSCATTERQPS(_0, _1) INSTR(vscatterqps, _0, _1)
#define VGATHERQPD(_0, _1) INSTR(vgatherqpd, _0, _1)
#define VSCATTERQPD(_0, _1) INSTR(vscatterqpd, _0, _1)

// Vector math

#define ADDPS(_0, _1, _2) INSTR(addps, _0, _1, _2)
#define ADDPD(_0, _1, _2) INSTR(addpd, _0, _1, _2)
#define SUBPS(_0, _1, _2) INSTR(subps, _0, _1, _2)
#define SUBPD(_0, _1, _2) INSTR(subpd, _0, _1, _2)
#define MULPS(_0, _1, _2) INSTR(mulps, _0, _1, _2)
#define MULPD(_0, _1, _2) INSTR(mulpd, _0, _1, _2)
#define XORPS(_0, _1) INSTR(xorps, _0, _1)
#define XORPD(_0, _1) INSTR(xorpd, _0, _1)
#define UCOMISS(_0, _1) INSTR(ucomiss, _0, _1)
#define UCOMISD(_0, _1) INSTR(ucomisd, _0, _1)
#define COMISS(_0, _1) INSTR(comiss, _0, _1)
#define COMISD(_0, _1) INSTR(comisd, _0, _1)

#define VADDSUBPS(_0, _1, _2) INSTR(vaddsubps, _0, _1, _2)
#define VADDSUBPD(_0, _1, _2) INSTR(vaddsubpd, _0, _1, _2)
#define VUCOMISS(_0, _1) INSTR(vucomiss, _0, _1)
#define VUCOMISD(_0, _1) INSTR(vucomisd, _0, _1)
#define VCOMISS(_0, _1) INSTR(vcomiss, _0, _1)
#define VCOMISD(_0, _1) INSTR(vcomisd, _0, _1)
#define VADDPS(_0, _1, _2) INSTR(vaddps, _0, _1, _2)
#define VADDPD(_0, _1, _2) INSTR(vaddpd, _0, _1, _2)
#define VSUBPS(_0, _1, _2) INSTR(vsubps, _0, _1, _2)
#define VSUBPD(_0, _1, _2) INSTR(vsubpd, _0, _1, _2)
#define VMULSS(_0, _1, _2) INSTR(vmulss, _0, _1, _2)
#define VMULSD(_0, _1, _2) INSTR(vmulsd, _0, _1, _2)
#define VMULPS(_0, _1, _2) INSTR(vmulps, _0, _1, _2)
#define VMULPD(_0, _1, _2) INSTR(vmulpd, _0, _1, _2)
#define VPMULLD(_0, _1, _2) INSTR(vpmulld, _0, _1, _2)
#define VPMULLQ(_0, _1, _2) INSTR(vpmullq, _0, _1, _2)
#define VPADDD(_0, _1, _2) INSTR(vpaddd, _0, _1, _2)
#define VPSLLD(_0, _1, _2) INSTR(vpslld, _0, _1, _2)
#define VXORPS(_0, _1, _2) INSTR(vxorps, _0, _1, _2)
#define VXORPD(_0, _1, _2) INSTR(vxorpd, _0, _1, _2)
#define VPXORD(_0, _1, _2) INSTR(vpxord, _0, _1, _2)
#define VFMADD132SS(_0, _1, _2) INSTR(vfmadd132ss, _0, _1, _2)
#define VFMADD213SS(_0, _1, _2) INSTR(vfmadd213ss, _0, _1, _2)
#define VFMADD231SS(_0, _1, _2) INSTR(vfmadd231ss, _0, _1, _2)
#define VFMADD132SD(_0, _1, _2) INSTR(vfmadd132sd, _0, _1, _2)
#define VFMADD213SD(_0, _1, _2) INSTR(vfmadd213sd, _0, _1, _2)
#define VFMADD231SD(_0, _1, _2) INSTR(vfmadd231sd, _0, _1, _2)
#define VFMADD132PS(_0, _1, _2) INSTR(vfmadd132ps, _0, _1, _2)
#define VFMADD213PS(_0, _1, _2) INSTR(vfmadd213ps, _0, _1, _2)
#define VFMADD231PS(_0, _1, _2) INSTR(vfmadd231ps, _0, _1, _2)
#define VFMADD132PD(_0, _1, _2) INSTR(vfmadd132pd, _0, _1, _2)
#define VFMADD213PD(_0, _1, _2) INSTR(vfmadd213pd, _0, _1, _2)
#define VFMADD231PD(_0, _1, _2) INSTR(vfmadd231pd, _0, _1, _2)

// Vector shuffles

#define PSHUFD(_0, _1, _2) INSTR(pshufd, _0, _1, _2)
#define SHUFPS(_0, _1, _2) INSTR(shufps, _0, _1, _2)
#define SHUFPD(_0, _1, _2) INSTR(shufpd, _0, _1, _2)

#define VSHUFPS(_0, _1, _2, _3) INSTR(vshufps, _0, _1, _2, _3)
#define VSHUFPD(_0, _1, _2, _3) INSTR(vshufpd, _0, _1, _2, _3)
#define VPERMILPS(_0, _1, _2) INSTR(vpermilps, _0, _1, _2)
#define VPERMILPD(_0, _1, _2) INSTR(vpermilpd, _0, _1, _2)
#define VPERM2F128(_0, _1, _2, _3) INSTR(vperm2f128, _0, _1, _2, _3)
#define VPERMPD(_0, _1, _2) INSTR(vpermpd, _0, _1, _2)
#define VUNPCKLPS(_0, _1, _2) INSTR(vunpcklps, _0, _1, _2)
#define VUNPCKHPS(_0, _1, _2) INSTR(vunpckhps, _0, _1, _2)
#define VUNPCKLPD(_0, _1, _2) INSTR(vunpcklpd, _0, _1, _2)
#define VUNPCKHPD(_0, _1, _2) INSTR(vunpckhpd, _0, _1, _2)
#define VSHUFF32X4(_0, _1, _2, _3) INSTR(vshuff32x4, _0, _1, _2, _3)
#define VSHUFF64X2(_0, _1, _2, _3) INSTR(vshuff64x2, _0, _1, _2, _3)
#define VINSERTF128(_0, _1, _2, _3) INSTR(vinsertf128, _0, _1, _2, _3)
#define VINSERTF32X4(_0, _1, _2, _3) INSTR(vinsertf32x4, _0, _1, _2, _3)
#define VINSERTF32X8(_0, _1, _2, _3) INSTR(vinsertf32x8, _0, _1, _2, _3)
#define VINSERTF64X2(_0, _1, _2, _3) INSTR(vinsertf64x2, _0, _1, _2, _3)
#define VINSERTF64X4(_0, _1, _2, _3) INSTR(vinsertf64x4, _0, _1, _2, _3)
#define VEXTRACTF128(_0, _1, _2) INSTR(vextractf128, _0, _1, _2)
#define VEXTRACTF32X4(_0, _1, _2) INSTR(vextractf32x4, _0, _1, _2)
#define VEXTRACTF32X8(_0, _1, _2) INSTR(vextractf32x8, _0, _1, _2)
#define VEXTRACTF64X2(_0, _1, _2) INSTR(vextractf64x4, _0, _1, _2)
#define VEXTRACTF64X4(_0, _1, _2) INSTR(vextractf64x4, _0, _1, _2)
#define VBLENDPS(_0, _1, _2, _3) INSTR(vblendps, _0, _1, _2, _3)
#define VBLENDPD(_0, _1, _2, _3) INSTR(vblendpd, _0, _1, _2, _3)
#define VBLENDMPS(_0, _1, _2) INSTR(vblendmps, _0, _1, _2)
#define VBLENDMPD(_0, _1, _2) INSTR(vblendmpd, _0, _1, _2)

// Prefetches

#define PREFETCH(_0, _1) INSTR(prefetcht##_0, _1)
#define PREFETCHW0(_0) INSTR(prefetchw, _0)
#define PREFETCHW1(_0) INSTR(prefetchwt1, _0)
#define VGATHERPFDPS(_0, _1) INSTR(vgatherpf##_0##dps, _1)
#define VSCATTERPFDPS(_0, _1) INSTR(vscatterpf##_0##dps, _1)
#define VGATHERPFDPD(_0, _1) INSTR(vgatherpf##_0##dpd, _1)
#define VSCATTERPFDPD(_0, _1) INSTR(vscatterpf##_0##dpd, _1)

// Mask operations

#ifdef __MIC__

#define KMOV(_0, _1) INSTR(kmov, _0, _1)
#define JKNZD(_0, _1) INSTR(jknzd, _0, _1)

#else

#define KMOV(_0, _1) INSTR(kmovw, _0, _1)
#define JKNZD(_0, _1) INSTR(kortestw, _0, _0) INSTR(jnz, _1)

#endif

#define KXNORW(_0, _1, _2) INSTR(kxnorw, _0, _1, _2)
#define KSHIFTRW(_0, _1, _2) INSTR(kshiftrw, _0, _1, _2)

// Other

#define RDTSC() INSTR(rdtsc)
#define VZEROALL() INSTR(vzeroall)
#define VZEROUPPER() INSTR(vzeroupper)

// Complex helper macros

#define CSCALE_AVX(x, alpha_r, alpha_i, tmp) \
    VPERMILPS(YMM(tmp), YMM(x), IMM(0xB1)) \
    VMULPS(YMM(x), YMM(x), YMM(alpha_r)) \
    VMULPS(YMM(tmp), YMM(tmp), YMM(alpha_i)) \
    VADDSUBPS(YMM(x), YMM(x), YMM(tmp))

#define ZSCALE_AVX(x, alpha_r, alpha_i, tmp) \
    VPERMILPD(YMM(tmp), YMM(x), IMM(0x5)) \
    VMULPD(YMM(x), YMM(x), YMM(alpha_r)) \
    VMULPD(YMM(tmp), YMM(tmp), YMM(alpha_i)) \
    VADDSUBPD(YMM(x), YMM(x), YMM(tmp))

// Transposes

// 00 01 02 03 04 05 06 07
// 10 11 12 13 14 15 16 17
// 20 21 22 23 24 25 26 27
// 30 31 32 33 34 35 36 37
// 40 41 42 43 44 45 46 47
// 50 51 52 53 54 55 56 57

// 00 10 02 12 04 14 06 16
// 20 30 22 32 24 34 26 36
// 01 11 03 13 05 15 07 17
// 21 31 23 33 25 35 27 37
// 40 50 42 52 44 54 46 56
// 41 51 43 53 45 55 47 57

// 00 10 20 30 04 14 24 34
// 02 12 22 32 06 16 26 36
// 01 11 21 31 05 15 25 35
// 03 13 23 33 07 17 27 37
// 40 50 42 52 44 54 46 56
// 41 51 43 53 45 55 47 57

#define TRANSPOSE_S8X6_AVX_ASM(x1, x2, x3, x4, x5, x6, \
                               y1, y2, y3, y4, y5, y6, y7, y8, \
                               t1, t2, t3, t4, t5, t6) \
    VUNPCKLPS(YMM(t1), YMM(x1), YMM(x2)) \
    VUNPCKLPS(YMM(t2), YMM(x3), YMM(x4)) \
    VUNPCKHPS(YMM(t3), YMM(x1), YMM(x2)) \
    VUNPCKHPS(YMM(t4), YMM(x3), YMM(x4)) \
    VSHUFPS(YMM(t5), YMM(t1), YMM(t2), IMM(0x4e)) \
    VBLENDPS(YMM(t1), YMM(t1), YMM(t5), IMM(0xcc)) \
    VBLENDPS(YMM(t2), YMM(t5), YMM(t2), IMM(0xcc)) \
    VSHUFPS(YMM(t5), YMM(t1), YMM(t2), IMM(0x4e)) \
    VBLENDPS(YMM(t3), YMM(t3), YMM(t5), IMM(0xcc)) \
    VBLENDPS(YMM(t4), YMM(t5), YMM(t4), IMM(0xcc)) \
    VUNPCKLPS(YMM(t5), YMM(x5), YMM(x6)) \
    VUNPCKHPS(YMM(t6), YMM(x5), YMM(x6)) \
    VPERM2F128(YMM(y1), YMM(t1), YMM(t3), IMM(0x20)) \
    VPERM2F128(YMM(y2), YMM(t2), YMM(t4), IMM(0x20)) \
    VPERM2F128(YMM(y3), YMM(t1), YMM(t3), IMM(0x31)) \
    VPERM2F128(YMM(y4), YMM(t2), YMM(t4), IMM(0x31))

// 00 01 02 03
// 10 11 12 13
// 20 21 22 23
// 30 31 32 33

// 00 01 02 03
// 01 11 03 13
// 20 21 22 23
// 21 31 23 33
// 00 10 02 12
// 20 30 22 32

// 00 10 20 30
// 01 11 03 13
// 02 12 22 32
// 21 31 23 33
// 03 13 21 31

// 00 10 20 30
// 01 11 21 31
// 02 12 22 32
// 03 13 23 33

#define TRANSPOSE_D4X4_AVX_ASM(x1, x2, x3, x4, \
                               t1, t2) \
    VUNPCKLPD(YMM(t1), YMM(x1), YMM(x2)) \
    VUNPCKHPD(YMM(x2), YMM(x1), YMM(x2)) \
    VUNPCKLPD(YMM(t2), YMM(x3), YMM(x4)) \
    VUNPCKHPD(YMM(x4), YMM(x3), YMM(x4)) \
    VINSERTF128(YMM(x1), YMM(t1), XMM(t2), IMM(1)) \
    VPERM2F128(YMM(x3), YMM(t1), YMM(t2), IMM(0x31)) \
    VPERM2F128(YMM(t1), YMM(x2), YMM(x4), IMM(0x21)) \
    VBLENDPD(YMM(x2), YMM(x2), YMM(t1), IMM(0xc)) \
    VBLENDPD(YMM(x4), YMM(x4), YMM(t1), IMM(0x3))

#define TRANSPOSE_D4X2_AVX_ASM(x1, x2, \
                               y1, y2, y3, y4) \
    VUNPCKLPD(YMM(y1), YMM(x1), YMM(x2)) \
    VUNPCKHPD(YMM(y2), YMM(x1), YMM(x2)) \
    VEXTRACTF128(XMM(y3), YMM(y1), IMM(0x1)) \
    VEXTRACTF128(XMM(y4), YMM(y2), IMM(0x1))

#define TRANSPOSE_D4X2_AVX_INT(x1, x2, \
                               y1, y2, y3, y4) \
    __m256d t1 = _mm256_shuffle_pd(x1, x2, 0x0); \
    __m256d t2 = _mm256_shuffle_pd(x1, x2, 0xf); \
    y1 = _mm256_extractf128_pd(t1, 0x0); \
    y2 = _mm256_extractf128_pd(t2, 0x0); \
    y3 = _mm256_extractf128_pd(t1, 0x1); \
    y4 = _mm256_extractf128_pd(t2, 0x1);

#endif
