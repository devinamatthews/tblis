AC_INIT(tblis, m4_esyscmd_s([cat version]), dmatthews@utexas.edu, tblis, http://www.github.com/devinamatthews/tblis)
AM_INIT_AUTOMAKE([foreign silent-rules subdir-objects])
AM_SILENT_RULES([yes])
AM_MAINTAINER_MODE([disable])

AC_CONFIG_MACRO_DIR([m4])

AC_CONFIG_HEADERS(config.h)

AC_PROG_CC([gcc clang icc])
AC_PROG_CC_C99
AM_PROG_AS
AC_PROG_EGREP
AC_PROG_AWK
AX_BLAS([blas_found=yes])
AC_PROG_CXX([g++ clang++ icpc])
AX_CXX_COMPILE_STDCXX_11([noext])
AC_LANG([C++])
AC_C_RESTRICT

AC_DEFINE([RESTRICT], [_tblis_restrict], [More convenient macro for restrict.])
AM_CONDITIONAL([ENABLE_BLAS], [test x"$blas_found" = xyes])

AC_SEARCH_LIBS([clock_gettime], [rt])

topdir="\"`(cd $srcdir && pwd)`\""
AC_DEFINE_UNQUOTED([TOPDIR], [$topdir], [The top source directory.])

#
# Check for memkind
#
AC_CHECK_HEADERS([hbwmalloc.h memkind.h])
AC_SEARCH_LIBS([hbw_malloc], [memkind])
                
#
# Determine configurations to build
#

AC_ARG_ENABLE([config],
[  --enable-config=...     a comma-separated list of configurations
                          @<:@default=auto@:>@
  
                          possible values are:
                          
                          armv7a, armv8a, bgq, bulldozer, carrizo, cortex-a15,
                          cortex-a9, dunnington, haswell, knl, loongson3a, mic,
                          piledriver, power7, reference, sandybridge, auto

                          the following meta-configurations are also available:
 
                          intel = dunnington,sandybridge,haswell,knl
                          arm = armv7a,armv8a,cortex-a9,cortex-a15
                          amd = bulldozer,piledriver,carrizo
                          x86 = intel,amd],
              [], [enable_config=auto])
                             
if echo $enable_config | grep -q auto; then
    enable_config=
    AC_CHECK_DEFINE([__MIC__], [enable_config=mic])
    AC_CHECK_DEFINE([__x86_64__], [enable_config=x86])
    AC_CHECK_DEFINE([_M_X64], [enable_config=x86])
    AC_CHECK_DEFINE([__i386], [enable_config=x86])
    AC_CHECK_DEFINE([_M_IX86], [enable_config=x86])
    AC_CHECK_DEFINE([__aarch64__], [enable_config=arm])
    AC_CHECK_DEFINE([__arm__], [enable_config=arm])
    AC_CHECK_DEFINE([_M_ARM], [enable_config=arm])
    AC_CHECK_DEFINE([__powerpc64__], [enable_config=power7])
    AC_CHECK_DEFINE([__ppc64__], [enable_config=power7])
    AC_CHECK_DEFINE([__PPC64__], [enable_config=power7])
    AC_CHECK_DEFINE([__powerpc__], [enable_config=power7])
    AC_CHECK_DEFINE([__ppc__], [enable_config=power7])
    AC_CHECK_DEFINE([__PPC__], [enable_config=power7])
    AC_CHECK_DEFINE([__bgq__], [enable_config=bgq])
    AC_CHECK_DEFINE([__mips], [enable_config=loongson3a])
fi
                             
if echo $enable_config | grep -qv reference; then
    enable_config="${enable_config},reference"
fi

configs=`echo $enable_config | sed 's/x86/intel,amd/' \
                             | sed 's/intel/dunnington,sandybridge,haswell,knl/' \
                             | sed 's/arm/armv7a,armv8a,cortex-a9,cortex-a15/' \
                             | sed 's/amd/bulldozer,piledriver,carrizo/' \
                             | sed 's/,/ /g'`
                             
AC_MSG_CHECKING([enabled configurations])
AC_MSG_RESULT([$configs])

AM_CONDITIONAL(ENABLE_ARMV7A, [echo $configs | grep -q armv7a])
AM_CONDITIONAL(ENABLE_ARMV8A, [echo $configs | grep -q armv8a])
AM_CONDITIONAL(ENABLE_BGQ, [echo $configs | grep -q bgq])
AM_CONDITIONAL(ENABLE_BULLDOZER, [echo $configs | grep -q bulldozer])
AM_CONDITIONAL(ENABLE_CARRIZO, [echo $configs | grep -q carrizo])
AM_CONDITIONAL(ENABLE_CORTEX_A15, [echo $configs | grep -q cortex-a15])
AM_CONDITIONAL(ENABLE_CORTEX_A9, [echo $configs | grep -q cortex-a9])
AM_CONDITIONAL(ENABLE_DUNNINGTON, [echo $configs | grep -q dunnington])
AM_CONDITIONAL(ENABLE_HASWELL, [echo $configs | grep -q haswell])
AM_CONDITIONAL(ENABLE_KNL, [echo $configs | grep -q knl])
AM_CONDITIONAL(ENABLE_LOONGSON3A, [echo $configs | grep -q loongson3a])
AM_CONDITIONAL(ENABLE_MIC, [echo $configs | grep -q mic])
AM_CONDITIONAL(ENABLE_PILEDRIVER, [echo $configs | grep -q piledriver])
AM_CONDITIONAL(ENABLE_POWER7, [echo $configs | grep -q power7])
AM_CONDITIONAL(ENABLE_SANDYBRIDGE, [echo $configs | grep -q sandybridge])

vendor_string=`$CC --version 2>/dev/null`
if test x"$vendor_string" = x; then
    vendor_string=`$CC -qversion 2>/dev/null`
fi
if test x"$vendor_string" = x; then
    AC_MSG_ERROR([Unable to determine compiler vendor.])
fi

cc_vendor=`echo $vendor_string | $EGREP -o 'icc|gcc|clang|emcc|pnacl|IBM' | { read first rest ; echo $first ; }`
if test x"$cc_vendor" = x; then
    AC_MSG_ERROR([Unable to determine compiler vendor.])
fi

AM_CONDITIONAL(ENABLE_INTEL_COMPILER, [test x"$cc_vendor" = xicc])

include_configs=""
for config in $configs; do
include_configs="$include_configs
#include \"configs/$config/config.hpp\""
done

AC_SUBST([INCLUDE_CONFIGS], [$include_configs])
AM_SUBST_NOTMAKE([INCLUDE_CONFIGS])

foreach_config=""
for config in $configs; do
foreach_config="$foreach_config
FOREACH_CONFIG(${config}_config)"
done

AC_SUBST([FOREACH_CONFIG], [$foreach_config])
AM_SUBST_NOTMAKE([FOREACH_CONFIG])

foreach_config_and_type=""
for config in $configs; do
foreach_config_and_type="$foreach_config_and_type
#define FOREACH_TYPE(type) \\
FOREACH_CONFIG_AND_TYPE(type, ${config}_config)
#include \"foreach_type.h\""
done

AC_SUBST([FOREACH_CONFIG_AND_TYPE], [$foreach_config_and_type])
AM_SUBST_NOTMAKE([FOREACH_CONFIG_AND_TYPE])

types="float double scomplex dcomplex"
foreach_type=""
for type in $types; do
foreach_type="$foreach_type
FOREACH_TYPE($type)"
done

AC_SUBST([FOREACH_TYPE], [$foreach_type])
AM_SUBST_NOTMAKE([FOREACH_TYPE])

LT_INIT
                
mkdir -p bin

AC_CONFIG_FILES([Makefile 
                 src/configs/foreach_type.h
                 src/configs/foreach_config.h
                 src/configs/foreach_config_and_type.h
                 src/configs/include_configs.hpp])
AX_PREFIX_CONFIG_H([tblis_config.h], [TBLIS])
AC_CONFIG_SUBDIRS([external/tci])
AC_OUTPUT

rm config.h
