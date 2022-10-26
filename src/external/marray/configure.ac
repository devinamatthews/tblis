AC_INIT(marray, 1.0, damatthews@smu.edu, marray, http://www.github.com/devinamatthews/marray)
AC_CONFIG_AUX_DIR([.])
AM_INIT_AUTOMAKE([foreign silent-rules subdir-objects])
AM_SILENT_RULES([yes])
AM_MAINTAINER_MODE([disable])

AC_CONFIG_MACRO_DIR([m4])

AC_PROG_CXX([g++ clang++ icpc])
AX_CXX_COMPILE_STDCXX([17], [noext])
AC_LANG([C++])

LT_INIT

mkdir -p bin

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
