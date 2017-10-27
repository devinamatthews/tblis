AC_INIT(tci, 1.0, dmatthews@utexas.edu, tblis, http://www.github.com/devinamatthews/tci)
AM_INIT_AUTOMAKE([foreign silent-rules subdir-objects])
AM_SILENT_RULES([yes])
AM_MAINTAINER_MODE([disable])

AC_CONFIG_MACRO_DIR([m4])

AC_PROG_CC([icc gcc clang])
AC_PROG_CC_C99
AC_PROG_CXX([icpc g++ clang++])
LT_INIT

#
# Get user preferences
#

AC_ARG_ENABLE([mutex], AS_HELP_STRING([--enable-mutex=type],
    [use a mutex of the specified type, one of auto, pthread_mutex, \
    pthread_spinlock, osx_spinlock, omp_lock, atomic_spinlock @<:@default=auto@:>@]),
    [mutex_type=$enable_mutex], [mutex_type=auto])

AC_ARG_ENABLE([barrier], AS_HELP_STRING([--enable-barrier=type],
    [use a barrier of the specified type, one of auto, pthread_barrier, \
    spin_barrier @<:@default=auto@:>@]),
    [barrier_type=$enable_barrier], [barrier_type=auto])

AC_ARG_ENABLE([thread-model], AS_HELP_STRING([--enable-thread-model@<:@=model@:>@],
    [enable threading with the specified model, one of none, auto, task, \
    openmp, pthreads, tbb, omptask, winthreads, ppl, or dispatch @<:@default=auto@:>@]),
    [thread_model=$enable_thread_model], [thread_model=auto])

AC_ARG_ENABLE([threading], AS_HELP_STRING([--enable-threading=@<:@=model@:>@],
    [alias for --enable-thread-model]), [thread_model=$enable_threading], [])
    
if test x"$thread_model" = xno; then
    thread_model=none
fi

#
# Check for pthreads
#

AX_PTHREAD([have_pthreads=yes], [have_pthreads=no])

have_pthread_mutex=no
have_pthread_spin=no
have_pthread_barrier=no
        
if test x$have_pthreads = xyes; then
    LIBS_save="$LIBS"
    CFLAGS_save="$CFLAGS"
    LIBS="$LIBS $PTHREAD_LIBS"
    CFLAGS="$CFLAGS $PTHREAD_CFLAGS"
    
    AC_CHECK_FUNCS([pthread_spin_init pthread_mutex_init pthread_barrier_init])
    
    if test x"$ac_cv_header_pthread_h" = xyes -a \
            x"$ac_cv_func_pthread_mutex_init" = xyes; then
        have_pthread_mutex=yes
    fi
    
    if test x"$ac_cv_header_pthread_h" = xyes -a \
            x"$ac_cv_func_pthread_spin_init" = xyes; then
        have_pthread_spin=yes
    fi
    
    if test x"$ac_cv_header_pthread_h" = xyes -a \
            x"$ac_cv_func_pthread_barrier_init" = xyes; then
        have_pthread_barrier=yes
    fi
    
    LIBS="$LIBS_save"
    CFLAGS="$CFLAGS_save"
fi

#
# Check for OpenMP
#

AC_OPENMP
CFLAGS_save="$CFLAGS"
CFLAGS="$CFLAGS $OPENMP_CFLAGS"
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
#include "omp.h"
#ifndef _OPENMP
#error
#endif
]])], [have_openmp="yes"], [have_openmp="no"])
CFLAGS="$CFLAGS_save"

#
# Check for TBB
#

AC_LANG_PUSH([C++])
LIBS_save="$LIBS"

AC_CHECK_HEADERS([tbb/tbb.h])
AC_SEARCH_LIBS([TBB_runtime_interface_version], [tbb])

if test x"$ac_cv_header_tbb_tbb_h" = xyes && \
   test x"$ac_cv_search_TBB_runtime_interface_version" != xno; then
    have_tbb=yes
    if test x"$ac_cv_search_TBB_runtime_interface_version" != "xnone required"; then
        TBB_LIBS="$ac_cv_search_TBB_runtime_interface_version"
    fi
else
    have_tbb=no
fi

LIBS="$LIBS_save"
AC_LANG_POP

#
# Check for libdispatch
#

LIBS_save="$LIBS"

AC_CHECK_HEADERS([dispatch/dispatch.h])
AC_SEARCH_LIBS([dispatch_async_f], [dispatch])

if test x"$ac_cv_header_dispatch_dispatch_h" = xyes && \
   test x"$ac_cv_search_dispatch_async_f" != xno; then
    have_gcd=yes
    if test x"$ac_cv_search_dispatch_async_f" != "xnone required"; then
        DISPATCH_LIBS="$ac_cv_search_dispatch_async_f"
    fi
else
    have_gcd=no
fi

LIBS="$LIBS_save"

#
# Check for Windows
#

AC_CHECK_HEADERS([windows.h])

if test x"$ac_cv_header_windows_h" = xyes; then
    have_windows=yes
else
    have_windows=no
fi

#
# Check for PPL
#

LIBS_save="$LIBS"

AC_CHECK_HEADERS([ppl.h])

if test x"$ac_cv_header_ppl_h" = xyes; then
    have_ppl=yes
else
    have_ppl=no
fi

#
# Check for OSSpinLock
#

AC_CHECK_HEADER([libkern/OSAtomic.h])
AC_CHECK_FUNCS([OSSpinLockLock])

if test x"$ac_cv_header_libkern_OSAtomic_h" = xyes -a \
        x"$ac_cv_func_OSSpinLockLock" = xyes; then
    have_osx_spinlock=yes
else
    have_osx_spinlock=no
fi

#
# Check for __atomic builtins
#

AX_GCC_BUILTIN(__atomic_load_n)
AX_GCC_BUILTIN(__atomic_load)
AX_GCC_BUILTIN(__atomic_store_n)
AX_GCC_BUILTIN(__atomic_store)
AX_GCC_BUILTIN(__atomic_exchange_n)
AX_GCC_BUILTIN(__atomic_exchange)
AX_GCC_BUILTIN(__atomic_compare_exchange_n)
AX_GCC_BUILTIN(__atomic_compare_exchange)
AX_GCC_BUILTIN(__atomic_add_fetch)
AX_GCC_BUILTIN(__atomic_sub_fetch)
AX_GCC_BUILTIN(__atomic_and_fetch)
AX_GCC_BUILTIN(__atomic_xor_fetch)
AX_GCC_BUILTIN(__atomic_or_fetch)
AX_GCC_BUILTIN(__atomic_nand_fetch)
AX_GCC_BUILTIN(__atomic_fetch_add)
AX_GCC_BUILTIN(__atomic_fetch_sub)
AX_GCC_BUILTIN(__atomic_fetch_and)
AX_GCC_BUILTIN(__atomic_fetch_xor)
AX_GCC_BUILTIN(__atomic_fetch_or)
AX_GCC_BUILTIN(__atomic_fetch_nand)
AX_GCC_BUILTIN(__atomic_test_and_set)
AX_GCC_BUILTIN(__atomic_clear)
AX_GCC_BUILTIN(__atomic_thread_fence)
AX_GCC_BUILTIN(__atomic_signal_fence)
AX_GCC_BUILTIN(__atomic_always_lock_free)
AX_GCC_BUILTIN(__atomic_is_lock_free)
if test x"$ax_cv_have___atomic_load_n" = xno -o \
        x"$ax_cv_have___atomic_load" = xno -o \
        x"$ax_cv_have___atomic_store_n" = xno -o \
        x"$ax_cv_have___atomic_store" = xno -o \
        x"$ax_cv_have___atomic_exchange_n" = xno -o \
        x"$ax_cv_have___atomic_exchange" = xno -o \
        x"$ax_cv_have___atomic_compare_exchange_n" = xno -o \
        x"$ax_cv_have___atomic_compare_exchange" = xno -o \
        x"$ax_cv_have___atomic_add_fetch" = xno -o \
        x"$ax_cv_have___atomic_sub_fetch" = xno -o \
        x"$ax_cv_have___atomic_and_fetch" = xno -o \
        x"$ax_cv_have___atomic_xor_fetch" = xno -o \
        x"$ax_cv_have___atomic_or_fetch" = xno -o \
        x"$ax_cv_have___atomic_nand_fetch" = xno -o \
        x"$ax_cv_have___atomic_fetch_add" = xno -o \
        x"$ax_cv_have___atomic_fetch_sub" = xno -o \
        x"$ax_cv_have___atomic_fetch_and" = xno -o \
        x"$ax_cv_have___atomic_fetch_xor" = xno -o \
        x"$ax_cv_have___atomic_fetch_or" = xno -o \
        x"$ax_cv_have___atomic_fetch_nand" = xno -o \
        x"$ax_cv_have___atomic_test_and_set" = xno -o \
        x"$ax_cv_have___atomic_clear" = xno -o \
        x"$ax_cv_have___atomic_thread_fence" = xno -o \
        x"$ax_cv_have___atomic_signal_fence" = xno -o \
        x"$ax_cv_have___atomic_always_lock_free" = xno -o \
        x"$ax_cv_have___atomic_is_lock_free" = xno; then
    AC_MSG_ERROR([__atomic builtins required])
fi

#
# Check for libatomic
#

AC_SEARCH_LIBS([__atomic_compare_exchange_16], [atomic])

#
# Determine mutex type
#

if test x"$mutex_type" = xauto; then
    if test x"$have_osx_spinlock" = xyes; then
        mutex_type=osx_spinlock
    elif test x"$have_pthread_spinlock" = xyes; then
        mutex_type=pthread_spinlock
    else
        mutex_type=atomic_spinlock
    fi
fi

USE_PTHREAD_MUTEX=0
USE_PTHREAD_SPINLOCK=0
USE_OSX_SPINLOCK=0
USE_OMP_LOCK=0
USE_ATOMIC_SPINLOCK=0
   
AC_MSG_CHECKING([mutex type]) 
if test x"$mutex_type" = xpthread_mutex; then
    if test x"$have_pthread_mutex" = xno; then
        AC_MSG_ERROR([pthread mutex requested but not available])
    fi
    USE_PTHREAD_MUTEX=1
elif test x"$mutex_type" = xpthread_spinlock; then
    if test x"$have_pthread_spinlock" = xno; then
        AC_MSG_ERROR([pthread spinlock requested but not available])
    fi
    USE_PTHREAD_SPINLOCK=1
elif test x"$mutex_type" = xosx_spinlock; then
    if test x"$have_osx_spinlock" = xno; then
        AC_MSG_ERROR([OSX OSSpinLock requested but not available])
    fi
    USE_OSX_SPINLOCK=1
elif test x"$mutex_type" = xomp_lock; then
    if test x"$have_openmp" = xno; then
        AC_MSG_ERROR([omp_lock requested but not available])
    fi
    USE_OMP_LOCK=1
elif test x"$mutex_type" = xatomic_spinlock; then
    USE_ATOMIC_SPINLOCK=1
else
    AC_MSG_ERROR([invalid mutex type specified])
fi
AC_MSG_RESULT([$mutex_type])

AC_SUBST([USE_PTHREAD_MUTEX])
AC_SUBST([USE_PTHREAD_SPINLOCK])
AC_SUBST([USE_OSX_SPINLOCK])
AC_SUBST([USE_OMP_LOCK])
AC_SUBST([USE_ATOMIC_SPINLOCK])

#
# Determine barrier type
#

if test x"$barrier_type" = xauto; then
    barrier_type=spin_barrier
fi

USE_PTHREAD_BARRIER=0
USE_SPIN_BARRIER=0
    
AC_MSG_CHECKING([barrier type]) 
if test x"$barrier_type" = xpthread_barrier; then
    if test x"$have_pthread_barrier" = xno; then
        AC_MSG_ERROR([pthread barrier requested but not available])
    fi
    USE_PTHREAD_BARRIER=1
elif test x"$barrier_type" = xspin_barrier; then
    USE_SPIN_BARRIER=1
else
    AC_MSG_ERROR([invalid barrier type specified])
fi
AC_MSG_RESULT([$barrier_type])

AC_SUBST([USE_PTHREAD_BARRIER])
AC_SUBST([USE_SPIN_BARRIER])

#
# Determine thread model
#

if test x"$thread_model" = xauto; then
    if test x"$have_openmp" = xyes; then
        thread_model=openmp
    elif test x"$have_pthreads" = xyes; then
        thread_model=pthread
    elif test x"$have_windows" = xyes; then
        thread_model=windows
    elif test x"$have_tbb" = xyes; then
        thread_model=tbb
    elif test x"$have_ppl" = xyes; then
        thread_model=ppl
    elif test x"$have_dispatch" = xyes; then
        thread_model=dispatch
    else
        thread_model=none
    fi
elif test x"$thread_model" = xtask; then
    if test x"$have_tbb" = xyes; then
        thread_model=tbb
    elif test x"$have_ppl" = xyes; then
        thread_model=ppl
    elif test x"$have_dispatch" = xyes; then
        thread_model=dispatch
    elif test x"$have_openmp" = xyes; then
        thread_model=omptask
    else
        thread_model=none
    fi
fi

USE_OPENMP_THREADS=0
USE_PTHREADS_THREADS=0
USE_TBB_THREADS=0
USE_DISPATCH_THREADS=0
USE_WINDOWS_THREADS=0
USE_PPL_THREADS=0
USE_OMPTASK_THREADS=0

AC_MSG_CHECKING([thread model]) 
if test x"$thread_model" = xopenmp; then
    if test x"$have_openmp" = xno; then
        AC_MSG_ERROR([openmp requested but not available])
    fi
    USE_OPENMP_THREADS=1
    CFLAGS="$CFLAGS $OPENMP_CFLAGS"
    CXXFLAGS="$CXXFLAGS $OPENMP_CFLAGS"
elif test x"$thread_model" = xomptask; then
    if test x"$have_openmp" = xno; then
        AC_MSG_ERROR([omptask requested but not available])
    fi
    USE_OMPTASK_THREADS=1
    CFLAGS="$CFLAGS $OPENMP_CFLAGS"
    CXXFLAGS="$CXXFLAGS $OPENMP_CFLAGS"
elif test x"$thread_model" = xpthread || \
     test x"$thread_model" = xpthreads; then
    if test x"$have_pthreads" = xno; then
        AC_MSG_ERROR([pthreads requested but not available])
    fi
    USE_PTHREADS_THREADS=1
    LIBS="$LIBS $PTHREAD_LIBS"
    CFLAGS="$CFLAGS $PTHREAD_CFLAGS"
elif test x"$thread_model" = xtbb; then
    if test x"$have_tbb" = xno; then
        AC_MSG_ERROR([tbb requested but not available])
    fi
    USE_TBB_THREADS=1
    LIBS="$LIBS $TBB_LIBS"
elif test x"$thread_model" = xdispatch || \
     test x"$thread_model" = xlibdispatch || \
     test x"$thread_model" = xgcd; then
    if test x"$have_dispatch" = xno; then
        AC_MSG_ERROR([dispatch requested but not available])
    fi
    USE_DISPATCH_THREADS=1
    LIBS="$LIBS $DISPATCH_LIBS"
elif test x"$thread_model" = xwindows || \
     test x"$thread_model" = xwinthreads; then
    if test x"$have_windows" = xno; then
        AC_MSG_ERROR([winthreads requested but not available])
    fi
    USE_WINDOWS_THREADS=1
elif test x"$thread_model" = xppl; then
    if test x"$have_windows" = xno || \
       test x"$have_ppl" = xno; then
        AC_MSG_ERROR([ppl requested but not available])
    fi
    USE_PPL_THREADS=1
elif test x"$thread_model" != xnone; then
    AC_MSG_ERROR([Invalid threading model specified.])
fi
AC_MSG_RESULT([$thread_model]) 

AC_SUBST([USE_OPENMP_THREADS])
AC_SUBST([USE_OMPTASK_THREADS])
AC_SUBST([USE_PTHREADS_THREADS])
AC_SUBST([USE_TBB_THREADS])
AC_SUBST([USE_PPL_THREADS])
AC_SUBST([USE_WINDOWS_THREADS])
AC_SUBST([USE_DISPATCH_THREADS])

AM_CONDITIONAL([CPLUSPLUS], [test x$USE_TBB_THREADS = x1 || \
                             test x$USE_PPL_THREADS = x1])

if test x$USE_TBB_THREADS = x1 || \
   test x$USE_PPL_THREADS = x1; then
    AX_CXX_COMPILE_STDCXX_11([noext])
fi

AC_CONFIG_FILES([Makefile tci/tci_config.h])
AC_OUTPUT
