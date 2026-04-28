AC_INIT(TCI,
    m4_normalize(m4_include([tci-version])),
    damatthews@smu.edu,
    tci,
    http://www.github.com/MatthewsResearchGroup/tci)

AC_CONFIG_MACRO_DIR([cmake])

AX_CMAKE_OPTION([static], AS_HELP_STRING([--enable-static], [Enable building a static library.]))

AX_CMAKE_OPTION([shared], AS_HELP_STRING([--disable-static], [Disable building a shared library.]))

AX_CMAKE_ENABLE([mutex], AS_HELP_STRING([--enable-mutex=type],
    [use a mutex of the specified type, one of auto, pthread_mutex, \
    pthread_spinlock, osx_spinlock, os_unfair_lock, omp_lock, atomic_spinlock @<:@default=auto@:>@]))

AX_CMAKE_ENABLE([barrier], AS_HELP_STRING([--enable-barrier=type],
    [use a barrier of the specified type, one of auto, pthread_barrier, \
    spin_barrier @<:@default=auto@:>@]))

AX_CMAKE_ENABLE([thread-model], AS_HELP_STRING([--enable-thread-model@<:@=model@:>@],
    [enable threading with the specified model, one of none, auto, task, \
    openmp, pthreads, tbb, omptask, winthreads, ppl, or dispatch @<:@default=auto@:>@]))

AX_CMAKE_ENABLE([threading], AS_HELP_STRING([--enable-threading=@<:@=model@:>@]))

AX_CMAKE()