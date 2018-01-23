dnl Copyright (C) 2002-2009 Xiph.org Foundation
dnl 
dnl Redistribution and use in source and binary forms, with or without
dnl modification, are permitted provided that the following conditions
dnl are met:
dnl 
dnl - Redistributions of source code must retain the above copyright
dnl notice, this list of conditions and the following disclaimer.
dnl 
dnl - Redistributions in binary form must reproduce the above copyright
dnl notice, this list of conditions and the following disclaimer in the
dnl documentation and/or other materials provided with the distribution.
dnl 
dnl - Neither the name of the Xiph.org Foundation nor the names of its
dnl contributors may be used to endorse or promote products derived from
dnl this software without specific prior written permission.
dnl 
dnl THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
dnl ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
dnl LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
dnl A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION
dnl OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
dnl SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
dnl LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
dnl DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
dnl THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
dnl (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
dnl OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

dnl as-gcc-inline-assembly.m4 0.1.0

dnl autostars m4 macro for detection of gcc inline assembly

dnl David Schleef <ds@schleef.org>

dnl $Id$

dnl AS_COMPILER_FLAG(ACTION-IF-ACCEPTED, [ACTION-IF-NOT-ACCEPTED])
dnl Tries to compile with the given CFLAGS.
dnl Runs ACTION-IF-ACCEPTED if the compiler can compile with the flags,
dnl and ACTION-IF-NOT-ACCEPTED otherwise.

AC_DEFUN([AS_GCC_INLINE_ASSEMBLY],
[
  AC_MSG_CHECKING([if compiler supports gcc-style inline assembly])

  AC_TRY_COMPILE([], [
#ifdef __GNUC_MINOR__
#if (__GNUC__ * 1000 + __GNUC_MINOR__) < 3004
#error GCC before 3.4 has critical bugs compiling inline assembly
#endif
#endif
__asm__ (""::) ], [flag_ok=yes], [flag_ok=no])

  if test "X$flag_ok" = Xyes ; then
    $1
    true
  else
    $2
    true
  fi
  AC_MSG_RESULT([$flag_ok])
])

AC_DEFUN([AC_TRY_ASSEMBLE],
[ac_c_ext=$ac_ext
 ac_ext=${ac_s_ext-s}
 cat > conftest.$ac_ext <<EOF
 .file "configure"
[$1]
EOF
if AC_TRY_EVAL(ac_compile); then
  ac_ext=$ac_c_ext
  ifelse([$2], , :, [  $2
  rm -rf conftest*])
else
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
  ac_ext=$ac_c_ext
ifelse([$3], , , [  rm -rf conftest*
  $3
])dnl
fi
rm -rf conftest*])


AC_DEFUN([AS_ASM_ARM_NEON],
[
  AC_MSG_CHECKING([if assembler supports NEON instructions on ARM])

  AC_TRY_ASSEMBLE([vorr d0,d0,d0], [flag_ok=yes], [flag_ok=no])

  if test "X$flag_ok" = Xyes ; then
    $1
    true
  else
    $2
    true
  fi
  AC_MSG_RESULT([$flag_ok])
])


AC_DEFUN([AS_ASM_ARM_MEDIA],
[
  AC_MSG_CHECKING([if assembler supports ARMv6 media instructions on ARM])

  AC_TRY_ASSEMBLE([shadd8 r3,r3,r3], [flag_ok=yes], [flag_ok=no])

  if test "X$flag_ok" = Xyes ; then
    $1
    true
  else
    $2
    true
  fi
  AC_MSG_RESULT([$flag_ok])
])


AC_DEFUN([AS_ASM_ARM_EDSP],
[
  AC_MSG_CHECKING([if assembler supports EDSP instructions on ARM])

  AC_TRY_ASSEMBLE([qadd r3,r3,r3], [flag_ok=yes], [flag_ok=no])

  if test "X$flag_ok" = Xyes ; then
    $1
    true
  else
    $2
    true
  fi
  AC_MSG_RESULT([$flag_ok])
])
