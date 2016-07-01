AC_INIT(tblis, 1.0, dmatthews@utexas.edu, tblis, http://www.github.com/devinamatthews/tblis)
AM_INIT_AUTOMAKE([foreign silent-rules subdir-objects])
LT_INIT
AM_SILENT_RULES([yes])
AM_MAINTAINER_MODE([disable])

AC_CONFIG_MACRO_DIR([m4])

AC_CONFIG_HEADERS(config.h)

AC_PROG_CC([gcc clang icc])
AM_PROG_AS
AC_PROG_CXX([g++ clang++ icpc])
AX_CXX_COMPILE_STDCXX_11([noext])
AC_LANG([C++])
AC_C_RESTRICT
AC_OPENMP

CPPFLAGS="$CPPFLAGS $OPENMP_CXXFLAGS"
LDFLAGS="$LDFLAGS $OPENMP_CXXFLAGS"

topdir="\"`(cd $srcdir && pwd)`\""
AC_DEFINE_UNQUOTED([TOPDIR], [$topdir], [The top source directory.])

AC_DEFINE([USE_CXX11_SPINLOCK], [1], [Use a spin-lock based on C++11 atomics])
AC_DEFINE([USE_SPIN_BARRIER], [1], [Use a spin-loop based barrier])
AC_DEFINE([USE_CXX11_THREADS], [1], [Use C++11 std::thread])

#
# Check for BLIS
#
AQ_WITH_PACKAGE([blis], [https://github.com/flame/blis],
                [], [.], [], [])
                
#
# Determine configurations to build
#

AC_ARG_ENABLE(config,
[  --enable-config         a comma-separated list of configurations
                          @<:@default=reference@:>@
  
                          possible values are:
                          
                          armv7a, armv8a, bgq, bulldozer, carrizo, cortex-a15,
                          cortex-a9, dunnington, haswell, knl, loongson3a, mic,
                          piledriver, power7, reference, sandybridge

                          the following meta-configurations are also available:
 
                          intel = dunnington,sandybridge,haswell,knl
                          arm = armv7a,armv8a,cortex-a9,cortex-a15
                          amd = bulldozer,piledriver,carrizo
                          x86 = intel,arm],
              [], [enable_config=reference])
              
configs=`echo $enable_config | sed 's/x86/intel,amd/' \
                             | sed 's/intel/dunnington,sandybridge,haswell,knl/' \
                             | sed 's/arm/armv7a,armv8a,cortex-a9,cortex-a15/' \
                             | sed 's/amd/bulldozer,piledriver,carrizo/' \
                             | sed 's/,/ /g'`
default_config=`echo $configs | { read first rest; echo $first; }`

AC_DEFINE_UNQUOTED([DEFAULT_CONFIG], [${default_config}_config], [Default configuration.])

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

AH_TOP([
#if defined(__MIC__)
#define ARCH_MIC 1
#elif defined(__INTEL_COMPILER) || defined(__i386__) || defined(__x86_64__) || \
      defined(_X86_) || defined(_M_IX86) || defined(_M_X64)
#define ARCH_INTEL 1
#elif defined(__arm__) || defined(_M_ARM) || defined(__aarch64__)
#define ARCH_ARM 1
#elif defined(_ARCH_PPC) || defined(_ARCH_PPC64)
#define ARCH_PPC 1
#endif
])

import_configs=""
for config in $configs; do
import_configs="$import_configs
#define TBLIS_CONFIG $config
#include \"tblis_import_config.hpp\""
done

AC_SUBST([IMPORT_CONFIGS], [$import_configs])
AM_SUBST_NOTMAKE([IMPORT_CONFIGS])

instantiate_types_for_config=""
for config in $configs; do
instantiate_types_for_config="$instantiate_types_for_config
#define INSTANTIATE_FOR_TYPE(type) \\
INSTANTIATE_FOR_CONFIG(type, ${config}_config)
#include \"tblis_instantiate_for_types.hpp\""
done

AC_SUBST([INSTANTIATE_TYPES_FOR_CONFIGS], [$instantiate_types_for_config])
AM_SUBST_NOTMAKE([INSTANTIATE_TYPES_FOR_CONFIGS])

instantiate_for_types=""
for type in float double scomplex dcomplex; do
instantiate_for_types="$instantiate_for_types
INSTANTIATE_FOR_TYPE($type)"
done

AC_SUBST([INSTANTIATE_FOR_TYPES], [$instantiate_for_types])
AM_SUBST_NOTMAKE([INSTANTIATE_FOR_TYPES])
                
mkdir -p bin

AC_CONFIG_FILES([Makefile 
                 src/tblis_import_configs.hpp
                 src/tblis_instantiate_for_configs.hpp
                 src/tblis_instantiate_for_types.hpp])
AX_PREFIX_CONFIG_H([tblis_config.hpp], [TBLIS])
AC_OUTPUT
