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
])
AH_BOTTOM([
#ifndef DEBUG
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

#
# Check for BLIS++
#
AQ_WITH_PACKAGE([blis++], [https://github.com/devinamatthews/blisplusplus],
                [blis++.hpp], [],
                [], [], [], [$blis_INCLUDES])
                
mkdir -p bin

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
