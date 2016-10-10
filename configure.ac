AC_INIT(tblis, 1.0, dmatthews@utexas.edu, tblis, http://www.github.com/devinamatthews/tblis)
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
# Determine threading options
#

AC_ARG_ENABLE([threading], AS_HELP_STRING([--enable-threading@<:@=model@:>@],
    [enable threading with the specified model, one of openmp, pthread, or c++11 @<:@default=openmp if available, or no threading when --enable-threading not specified@:>@]),
    [thread_model=$enable_threading], [thread_model=no])

if test x"$thread_model" = xyes; then
    thread_model=auto
fi
    
if test x"$thread_model" = xopenmp -o x"$thread_model" = xauto; then
    AC_OPENMP
    CPPFLAGS="$CPPFLAGS $OPENMP_CXXFLAGS"
    LDFLAGS="$LDFLAGS $OPENMP_CXXFLAGS"
    AC_CHECK_HEADER([omp.h], [thread_model=openmp],
    [
        if test x"$thread_model" = xopenmp; then
            AC_MSG_ERROR([OpenMP requested but not available.])
        fi
    ])
fi
    
if test x"$thread_model" = xpthread -o x"$thread_model" = xauto; then
    CPPFLAGS="$CPPFLAGS -pthread"
    LDFLAGS="$LDFLAGS -pthread"
    AC_CHECK_HEADER([pthread.h], [thread_model=pthread],
    [
        if test x"$thread_model" = xpthread; then
            AC_MSG_ERROR([pthreads requested but not available.])
        fi
    ])
fi

if test x"$thread_model" = xauto; then
    thread_model=c++11
fi

if test x"$thread_model" = xopenmp; then
    AC_DEFINE([USE_OPENMP_THREADS], [1], [Use OpenMP for threading])
elif test x"$thread_model" = xpthread; then
    AC_DEFINE([USE_PTHREAD_THREADS], [1], [Use pthreads for threading])
elif test x"$thread_model" = xc++11; then
    AC_DEFINE([USE_CXX11_THREADS], [1], [Use C++11 std::thread])
elif test x"$thread_model" != xno; then
    AC_MSG_ERROR([Invalid threading model specified.])
fi

AC_ARG_WITH([mutex], AS_HELP_STRING([--with-mutex=type],
    [use a mutex of the specified type, one of pthread_mutex, pthread_spinlock, osspinlock, std_mutex, omp_lock, or spinlock @<:@default=spinlock@:>@]),
    [mutex_type=$with_mutex], [mutex_type=spinlock])
    
if test x"$mutex_type" = xpthread_mutex; then
    AC_DEFINE([USE_PTHREAD_MUTEX], [1], [Use pthread_mutex for locking])
elif test x"$mutex_type" = xpthread_spinlock; then
    AC_DEFINE([USE_PTHREAD_SPINLOCK], [1], [Use pthread_spinlock for locking])
elif test x"$mutex_type" = xosspinlock; then
    AC_DEFINE([USE_OSX_SPINLOCK], [1], [Use OSSpinLock for locking])
elif test x"$mutex_type" = xstd_mutex; then
    AC_DEFINE([USE_CXX11_MUTEX], [1], [Use C++11 std::mutex for locking])
elif test x"$mutex_type" = xomp_lock; then
    AC_DEFINE([USE_OMP_LOCK], [1], [Use omp_lock for locking])
elif test x"$mutex_type" = xspinlock; then
    AC_DEFINE([USE_CXX11_SPINLOCK], [1], [Use a spin-lock based on C++11 atomics])
else
    AC_MSG_ERROR([Invalid mutex type specified.])
fi

AC_ARG_WITH([barrier], AS_HELP_STRING([--with-barrier=type],
    [use a barrier of the specified type, one of pthread, c++11, spin @<:@default=spin@:>@]),
    [barrier_type=$with_barrier], [barrier_type=spin])
    
if test x"$barrier_type" = xpthread; then
    AC_DEFINE([USE_PTHREAD_BARRIER], [1], [Use pthread_barrier for thread barriers])
elif test x"$barrier_type" = xc++11; then
    AC_DEFINE([USE_CXX11_BARRIER], [1], [Use a thread barrier based on C++11 condition variables])
elif test x"$barrier_type" = xspin; then
    AC_DEFINE([USE_SPIN_BARRIER], [1], [Use a spin-loop based barrier])
else
    AC_MSG_ERROR([Invalid barrier type specified.])
fi
                
#
# Determine configurations to build
#

AC_ARG_ENABLE([config],
[  --enable-config=...     a comma-separated list of configurations
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

AM_CONDITIONAL(IS_OSX, [test `uname -s` = Darwin])

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

AH_TOP([
#if defined(__MIC__)
#define ARCH_MIC 1
#elif defined(__ia64) || defined(__itanium__) || defined(_M_IA64)
#define ARCH_IA64 1
#elif defined(__x86_64__) || defined(_M_X64)
#define ARCH_X64 1
#elif defined(__i386) || defined(_M_IX86)
#define ARCH_X86 1
#elif defined(__aarch64__)
#define ARCH_ARM64 1
#elif defined(__arm__) || defined(_M_ARM)
#define ARCH_ARM32 1
#elif defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
#define ARCH_PPC64 1
#elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__)
#define ARCH_PPC32 1
#elif defined(__bgq__)
#define ARCH_BGQ 1
#elif defined(__sparc)
#define ARCH_SPARC
#elif defined(__mips)
#define ARCH_MIPS
#else
#error "Unknown architecture"
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

LT_INIT
                
mkdir -p bin

AC_CONFIG_FILES([Makefile 
                 src/configs/tblis_instantiate_for_configs.hpp
                 src/configs/tblis_instantiate_for_types.hpp])
AX_PREFIX_CONFIG_H([tblis_config.hpp], [TBLIS])
AC_OUTPUT

rm config.h
