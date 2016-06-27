AC_INIT(tblis, 1.0, dmatthews@utexas.edu, tensor, http://www.github.com/devinamatthews/tblis)
AM_INIT_AUTOMAKE([foreign silent-rules subdir-objects])
LT_INIT
AM_SILENT_RULES([yes])
AM_MAINTAINER_MODE([disable])

AC_CONFIG_MACRO_DIR([m4])

AM_CONFIG_HEADER(config.h)
AH_TOP([
#ifndef _CONFIG_H_
#define _CONFIG_H_

#if defined(__MIC__)
#define TBLIS_ARCH_MIC 1
#elif defined(__INTEL_COMPILER) || defined(__i386__) || defined(__x86_64__) || \
      defined(_X86_) || defined(_M_IX86) || defined(_M_X64)
#define TBLIS_ARCH_INTEL 1
#elif defined(__arm__) || defined(_M_ARM) || defined(__aarch64__)
#define TBLIS_ARCH_ARM 1
#elif defined(_ARCH_PPC) || defined(_ARCH_PPC64)
#define TBLIS_ARCH_PPC 1
#endif

])
AH_BOTTOM([

#ifndef TBLIS_DEBUG
#define NDEBUG
#endif

#endif
])

AC_PROG_CXX([g++ clang++ icpc])
AX_CXX_COMPILE_STDCXX_11([noext])
AC_LANG([C++])
AC_C_RESTRICT
AC_OPENMP

CPPFLAGS="$CPPFLAGS $OPENMP_CXXFLAGS"
LDFLAGS="$LDFLAGS $OPENMP_CXXFLAGS"

topdir="\"`(cd $srcdir && pwd)`\""
AC_DEFINE_UNQUOTED([TOPDIR], [$topdir], [The top source directory.])

#
# Check for BLIS
#
AQ_WITH_PACKAGE([blis], [https://github.com/flame/blis],
                [blis/blis.h], [],
                [bli_init], [-lblis])
                
mkdir -p bin

AC_CONFIG_FILES([Makefile])
AX_PREFIX_CONFIG_H([tblis_config.h], [TBLIS])
AC_OUTPUT
