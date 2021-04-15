AC_INIT(stl_ext, 1.0, dmatthews@utexas.edu, aquarius, http://www.github.com/devinamatthews/stl_ext)
AM_INIT_AUTOMAKE([foreign silent-rules subdir-objects])
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

#
# Configure Google Test
#
AQ_WITH_PACKAGE([GTest], [],
                [gtest/gtest.h], [],
                [testing::Test], [-lgtest -lgtest_main -lpthread])
                
mkdir -p bin

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
