cmake_minimum_required(VERSION 3.23)

file(STRINGS tci-version TCI_VERSION_RAW)
set(TCI_VERSION "${TCI_VERSION_RAW}")
string(REPLACE "." ";" TCI_VERSION_RAW ${TCI_VERSION_RAW})
list(GET TCI_VERSION_RAW 0 TCI_VERSION_MAJOR)
list(GET TCI_VERSION_RAW 1 TCI_VERSION_MINOR)
set(prefix ${INSTALL_PREFIX})

project(TCI
    VERSION "${TCI_VERSION}"
    LANGUAGES C CXX
    HOMEPAGE_URL "http://www.github.com/MatthewsResearchGroup/tci"
    DESCRIPTION "Thread Control Interface"
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ConfigureWrapper)
include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckCSourceCompiles)
include(CMakePackageConfigHelpers)

#
# Set up options
#

option(ENABLE_SHARED "build a shared library" ON)

option(ENABLE_STATIC "build a static library" OFF)

set(ENABLE_MUTEX auto CACHE STRING "use a mutex of the specified type, one of auto, pthread_mutex, \
pthread_spinlock, osx_spinlock, os_unfair_lock, omp_lock, atomic_spinlock [default=auto]")

set(ENABLE_BARRIER auto CACHE STRING "use a barrier of the specified type, one of auto, pthread_barrier, \
spin_barrier [default=auto]")

set(ENABLE_THREAD_MODEL auto CACHE STRING "enable threading with the specified model, one of none, auto, task, \
openmp, pthreads, tbb, omptask, winthreads, ppl, or dispatch [default=auto]")

set(ENABLE_THREADING auto CACHE STRING "synonym for ENABLE_THREAD_MODEL")

if(ENABLE_THREAD_MODEL STREQUAL auto AND NOT ENABLE_THREADING STREQUAL auto)
    set(ENABLE_THREAD_MODEL ${ENABLE_THREADING})
endif()

# Default to disabled for all mutex/barrier/thread options
set(USE_PTHREAD_MUTEX 0)
set(USE_PTHREAD_SPINLOCK 0)
set(USE_OSX_SPINLOCK 0)
set(USE_OMP_LOCK 0)
set(USE_ATOMIC_SPINLOCK 0)

set(USE_PTHREAD_BARRIER 0)
set(USE_SPIN_BARRIER 0)

set(USE_PTHREADS_THREADS 0)
set(USE_OPENMP_THREADS 0)
set(USE_TBB_THREADS 0)
set(USE_PPL_THREADS 0)
set(USE_OMPTASK_THREADS 0)
set(USE_DISPATCH_THREADS 0)
set(USE_WINDOWS_THREADS 0)

#
# Check for various libraries
#

#
# Check if we need -lm
#
check_symbol_exists(sqrt math.h HAVE_SQRT)
find_library(HAVE_LIBM m)
if(NOT HAVE_SQRT)
    if (NOT HAVE_LIBM)
        message(FATAL_ERROR "no math library found.")
    else()
        list(APPEND PRIVATE_LIBS ${HAVE_LIBM})
    endif()
endif()

# Detect pthread support
find_package(Threads)

# Check for OpenMP support
find_package(OpenMP)

# Check for Apple GCD (libdispatch on BSD)
check_include_file(dispatch/dispatch.h HAVE_DISPATCH_H)
check_library_exists(dispatch dispatch_group_async_f "" HAVE_DISPATCH_LIB)

# Check for TBB
find_package(TBB QUIET)

# Check for Windows/PPL
check_include_file(windows.h HAVE_WINDOWS_H)
check_include_file(ppl.h HAVE_PPL_H)

# Check for OSX/macOS spin locks
check_include_file(os/lock.h HAVE_OS_LOCK_H)
check_symbol_exists(os_unfair_lock_lock os/lock.h HAVE_OS_UNFAIR_LOCK_LOCK)
check_include_file(libkern/OSAtomic.h HAVE_LIBKERN_OSATOMIC_H)
check_symbol_exists(OSSpinLockLock libkern/OSAtomic.h HAVE_OSSPINLOCKLOCK)
check_library_exists(atomic __atomic_compare_exchange_16 "" HAVE_ATOMIC_LIB)

# Verify availability of GCC style __atomic builtins
check_c_source_compiles("
    int main()
    {
        // Test all atomic builtin functions
        volatile int x;
        int y, z;
        volatile char f;
        __atomic_load_n(&x, __ATOMIC_SEQ_CST);
        __atomic_load(&x, &y, __ATOMIC_SEQ_CST);
        __atomic_store_n(&x, 1, __ATOMIC_SEQ_CST);
        __atomic_store(&x, &y, __ATOMIC_SEQ_CST);
        __atomic_exchange_n(&x, 1, __ATOMIC_SEQ_CST);
        __atomic_exchange(&x, &y, &z, __ATOMIC_SEQ_CST);
        __atomic_compare_exchange_n(&x, &y, 0, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
        __atomic_compare_exchange(&x, &y, &z, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
        __atomic_add_fetch(&x, 1, __ATOMIC_SEQ_CST);
        __atomic_sub_fetch(&x, 1, __ATOMIC_SEQ_CST);
        __atomic_and_fetch(&x, 1, __ATOMIC_SEQ_CST);
        __atomic_xor_fetch(&x, 1, __ATOMIC_SEQ_CST);
        __atomic_or_fetch(&x, 1, __ATOMIC_SEQ_CST);
        __atomic_nand_fetch(&x, 1, __ATOMIC_SEQ_CST);
        __atomic_fetch_add(&x, 1, __ATOMIC_SEQ_CST);
        __atomic_fetch_sub(&x, 1, __ATOMIC_SEQ_CST);
        __atomic_fetch_and(&x, 1, __ATOMIC_SEQ_CST);
        __atomic_fetch_xor(&x, 1, __ATOMIC_SEQ_CST);
        __atomic_fetch_or(&x, 1, __ATOMIC_SEQ_CST);
        __atomic_fetch_nand(&x, 1, __ATOMIC_SEQ_CST);
        __atomic_test_and_set(&f, __ATOMIC_SEQ_CST);
        __atomic_clear(&f, __ATOMIC_SEQ_CST);
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
        __atomic_always_lock_free(sizeof(x), &x);
        __atomic_is_lock_free(sizeof(x), &x);
        return 0;
    }"
    TCI_HAVE_GCC_ATOMIC_BUILTINS
)
set(CMAKE_REQUIRED_FLAGS -std=c11)
check_c_source_compiles("
    #include <stdatomic.h>
    int main()
    {
        // Test C11 atomic functions
        _Atomic int x;
        int y;
        atomic_load_explicit(&x, memory_order_seq_cst);
        atomic_store_explicit(&x, 1, memory_order_seq_cst);
        atomic_exchange_explicit(&x, 1, memory_order_seq_cst);
        atomic_compare_exchange_strong_explicit(&x, &y, 0, memory_order_seq_cst, memory_order_seq_cst);
        atomic_compare_exchange_weak_explicit(&x, &y, 0, memory_order_seq_cst, memory_order_seq_cst);
        atomic_fetch_add_explicit(&x, 1, memory_order_seq_cst);
        atomic_fetch_sub_explicit(&x, 1, memory_order_seq_cst);
        atomic_fetch_and_explicit(&x, 1, memory_order_seq_cst);
        atomic_fetch_or_explicit(&x, 1, memory_order_seq_cst);
        atomic_fetch_xor_explicit(&x, 1, memory_order_seq_cst);
        atomic_flag_test_and_set_explicit((atomic_flag*)0, memory_order_seq_cst);
        atomic_flag_clear_explicit((atomic_flag*)0, memory_order_seq_cst);
        atomic_thread_fence(memory_order_seq_cst);
        atomic_signal_fence(memory_order_seq_cst);
        atomic_is_lock_free(&x);
        return 0;
    }"
    TCI_HAVE_C11_ATOMIC
)
unset(CMAKE_REQUIRED_FLAGS)

if(CMAKE_USE_PTHREADS_INIT)
    check_symbol_exists(pthread_mutex_init pthread.h HAVE_PTHREAD_MUTEX_INIT)
    check_symbol_exists(pthread_spin_init pthread.h HAVE_PTHREAD_SPIN_INIT)
    check_symbol_exists(pthread_barrier_init pthread.h HAVE_PTHREAD_BARRIER_INIT)
endif()

# Detect pthreads
if(CMAKE_USE_PTHREADS_INIT)
    set(HAVE_PTHREADS_THREADS 1)
endif()

# Detect OpenMP
if(OpenMP_C_FOUND)
    set(HAVE_OPENMP_THREADS 1)
    # This really should be check specifically
    set(HAVE_OMPTASK_THREADS 1)
    set(HAVE_OMP_LOCK 1)
endif()

# Detect libdispatch (Apple GCD)
if(HAVE_DISPATCH_H AND (HAVE_DISPATCH_LIB OR APPLE))
    set(HAVE_DISPATCH_THREADS 1)
endif()

# Detect TBB
if(TBB_FOUND)
    set(HAVE_TBB_THREADS 1)
endif()

# Detect Windows
if(WIN32 AND HAVE_WINDOWS_H)
    set(HAVE_WINDOWS_THREADS 1)
endif()

# Detect PPL
if(WIN32 AND HAVE_WINDOWS_H AND HAVE_PPL_H)
    set(HAVE_PPL_THREADS 1)
endif()

# Detect pthread_mutex
if(CMAKE_USE_PTHREADS_INIT AND HAVE_PTHREAD_MUTEX_INIT)
    set(HAVE_PTHREAD_MUTEX 1)
endif()

# Detect pthread_spin
if(CMAKE_USE_PTHREADS_INIT AND HAVE_PTHREAD_SPIN_INIT)
    set(HAVE_PTHREAD_SPINLOCK 1)
endif()

# Detect pthread_barrier
if(CMAKE_USE_PTHREADS_INIT AND HAVE_PTHREAD_BARRIER_INIT)
    set(HAVE_PTHREAD_BARRIER 1)
endif()

# Detect os_unfair_lock
if(HAVE_OS_LOCK_H AND HAVE_OS_UNFAIR_LOCK_LOCK)
    set(HAVE_OS_UNFAIR_LOCK 1)
endif()

# Detect OSSpinLock
if(HAVE_LIBKERN_OSATOMIC_H AND HAVE_OSSPINLOCKLOCK)
    set(HAVE_OSX_SPINLOCK 1)
endif()

#
# Determine mutex type
#

if(ENABLE_MUTEX STREQUAL auto)
    if(HAVE_OS_UNFAIR_LOCK)
        set(ENABLE_MUTEX os_unfair_lock)
    elseif(HAVE_OSX_SPINLOCK)
        set(ENABLE_MUTEX osx_spinlock)
    elseif(HAVE_PTHREAD_SPINLOCK)
        set(ENABLE_MUTEX pthread_spinlock)
    else()
        set(ENABLE_MUTEX atomic_spinlock)
    endif()
endif()
message(STATUS "mutex type selected: ${ENABLE_MUTEX}")

if(ENABLE_MUTEX STREQUAL pthread_mutex)
    if(NOT HAVE_PTHREAD_MUTEX)
        message(FATAL_ERROR "pthread mutex requested but not available")
    endif()
    set(USE_PTHREAD_MUTEX 1)
elseif(ENABLE_MUTEX STREQUAL pthread_spinlock)
    if(NOT HAVE_PTHREAD_SPINLOCK)
        message(FATAL_ERROR "pthread spinlock requested but not available")
    endif()
    set(USE_PTHREAD_SPINLOCK 1)
elseif(ENABLE_MUTEX STREQUAL os_unfair_lock)
    if(NOT HAVE_OS_UNFAIR_LOCK)
        message(FATAL_ERROR "macOS os_unfair_lock requested but not available")
    endif()
    set(USE_OS_UNFAIR_LOCK 1)
elseif(ENABLE_MUTEX STREQUAL osx_spinlock)
    if(NOT HAVE_OSX_SPINLOCK)
        message(FATAL_ERROR "OSX OSSpinLock requested but not available")
    endif()
    set(USE_OSX_SPINLOCK 1)
elseif (ENABLE_MUTEX STREQUAL omp_lock)
    if(NOT HAVE_OMP_LOCK)
        message(FATAL_ERROR "omp_lock requested but not available")
    endif()
    set(USE_OMP_LOCK 1)
elseif (ENABLE_MUTEX STREQUAL atomic_spinlock)
    set(USE_ATOMIC_SPINLOCK 1)
else()
    message(FATAL_ERROR "invalid mutex type specified")
endif()

#
# Determine barrier type
#

if(ENABLE_BARRIER STREQUAL auto)
    set(ENABLE_BARRIER spin_barrier)
endif()

if(ENABLE_BARRIER STREQUAL pthread_barrier)
    if(NOT HAVE_PTHREAD_BARRIER)
        message(FATAL_ERROR "pthread barrier requested but not available")
    endif()
    set(USE_PTHREAD_BARRIER 1)
elseif(ENABLE_BARRIER STREQUAL spin_barrier)
    set(USE_SPIN_BARRIER 1)
else()
    message(FATAL_ERROR "invalid barrier type specified")
endif()
message(STATUS "barrier type selected: ${ENABLE_BARRIER}")

#
# Determine thread model
#

if(ENABLE_THREAD_MODEL STREQUAL auto)
    if(HAVE_OPENMP_THREADS)
        set(ENABLE_THREAD_MODEL openmp)
    elseif(HAVE_PTHREADS_THREADS)
        set(ENABLE_THREAD_MODEL pthread)
    elseif(HAVE_WINDOWS_THREADS)
        set(ENABLE_THREAD_MODEL windows)
    elseif(HAVE_TBB_THREADS)
        set(ENABLE_THREAD_MODEL tbb)
    elseif(HAVE_PPL_THREADS)
        set(ENABLE_THREAD_MODEL ppl)
    elseif(HAVE_DISPATCH_THREADS)
        set(ENABLE_THREAD_MODEL dispatch)
    else()
        set(ENABLE_THREAD_MODEL none)
    endif()
elseif(ENABLE_THREAD_MODEL STREQUAL task)
    if(HAVE_TBB_THREADS)
        set(ENABLE_THREAD_MODEL tbb)
    elseif(HAVE_PPL_THREADS)
        set(ENABLE_THREAD_MODEL ppl)
    elseif(HAVE_DISPATCH_THREADS)
        set(ENABLE_THREAD_MODEL dispatch)
    elseif(HAVE_OMPTASK_THREADS)
        set(ENABLE_THREAD_MODEL omptask)
    else()
        set(ENABLE_THREAD_MODEL none)
    endif()
endif()
message(STATUS "thread model selected: ${ENABLE_THREAD_MODEL}")

if(ENABLE_THREAD_MODEL STREQUAL openmp)
    if(NOT HAVE_OPENMP_THREADS)
        message(FATAL_ERROR "openmp requested but not available")
    endif()
    set(USE_OPENMP_THREADS 1)
elseif(ENABLE_THREAD_MODEL STREQUAL omptask)
    if(NOT HAVE_OMPTASK_THREADS)
        message(FATAL_ERROR "omptask requested but not available")
    endif()
    set(USE_OMPTASK_THREADS 1)
elseif(ENABLE_THREAD_MODEL STREQUAL pthread OR
       ENABLE_THREAD_MODEL STREQUAL pthreads)
    if(NOT HAVE_PTHREADS_THREADS)
        message(FATAL_ERROR "pthreads requested but not available")
    endif()
    set(USE_PTHREADS_THREADS 1)
elseif(ENABLE_THREAD_MODEL STREQUAL tbb)
    if(NOT HAVE_TBB_THREADS)
        message(FATAL_ERROR "tbb requested but not available")
    endif()
    set(USE_TBB_THREADS 1)
elseif(ENABLE_THREAD_MODEL STREQUAL dispatch OR
       ENABLE_THREAD_MODEL STREQUAL libdispatch OR
       ENABLE_THREAD_MODEL STREQUAL gcd)
    if(NOT HAVE_DISPATCH_THREADS)
        message(FATAL_ERROR "dispatch requested but not available")
    endif()
    set(USE_DISPATCH_THREADS 1)
elseif(ENABLE_THREAD_MODEL STREQUAL windows OR
       ENABLE_THREAD_MODEL STREQUAL winthreads)
    if(NOT HAVE_WINDOWS_THREADS)
        message(FATAL_ERROR "winthreads requested but not available")
    endif()
    set(USE_WINDOWS_THREADS 1)
elseif(ENABLE_THREAD_MODEL STREQUAL ppl)
    if(NOT HAVE_PPL_THREADS)
        message(FATAL_ERROR "ppl requested but not available")
    endif()
    set(USE_PPL_THREADS 1)
elseif(NOT ENABLE_THREAD_MODEL STREQUAL none)
    message(FATAL_ERROR "Invalid threading model specified.")
endif()

#
# Determine required libraries and/or flags
#

if(HAVE_ATOMIC_LIB)
    list(APPEND PRIVATE_LIBS atomic)
    list(APPEND PKGCONFIG_LIBS -latomic)
endif()

if(USE_PTHREADS_THREADS OR USE_PTHREAD_MUTEX OR USE_PTHREAD_BARRIER OR USE_PTHREAD_SPINLOCK)
    list(APPEND PRIVATE_LIBS Threads::Threads)
    # This is probably not always correct
    list(APPEND PKGCONFIG_LIBS -lpthread)
    if(ENABLE_STATIC)
        set(THREADS_DEPENDENCY "find_dependency(Threads)")
    endif()
endif()

set(TCI_OPENMP_ENABLED 0)
if(USE_OPENMP_THREADS OR USE_OMP_LOCK)
    add_compile_options("${OpenMP_C_FLAGS}")
    add_link_options("${OpenMP_C_FLAGS}")
    list(APPEND PRIVATE_LIBS OpenMP::OpenMP_C)
    list(APPEND PKGCONFIG_LIBS "${OpenMP_C_FLAGS}")
    set(TCI_OPENMP_ENABLED 1)
endif()

if(USE_DISPATCH_THREADS AND HAVE_DISPATCH_LIB)
    list(APPEND PRIVATE_LIBS dispatch)
    list(APPEND PKGCONFIG_LIBS -ldispatch)
endif()

if(USE_TBB_THREADS)
    list(APPEND PRIVATE_LIBS TBB::tbb)
    list(APPEND PKGCONFIG_REQUIRES tbb)
    if(ENABLE_STATIC)
        set(TBB_DEPENDENCY "find_dependency(TBB)")
    endif()
endif()

if(USE_WINDOWS_THREADS)
    # no additional libraries required
endif()

if(USE_PPL_THREADS)
    # no additional libraries required
endif()

# Have to compile communicator and task_set with C++ if using TBB or PPL
if(USE_PPL_THREADS OR USE_TBB_THREADS)
    set(TCI_CXX ON)
endif()

string(REPLACE ";" " " PKGCONFIG_LIBS "${PKGCONFIG_LIBS}")
string(REPLACE ";" " " PKGCONFIG_REQUIRES "${PKGCONFIG_REQUIRES}")

#
# Configure TCI
#

configure_file(include/tci/tci_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/tci/tci_config.h @ONLY)
configure_file(tci.pc.in ${CMAKE_CURRENT_BINARY_DIR}/tci.pc @ONLY)
configure_file(TCIConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/TCIConfig.cmake @ONLY)

set(TCI_SOURCES
    src/tci/barrier.c
    src/tci/communicator.c
    src/tci/task_set.c
    src/tci/context.c
    src/tci/mutex.c
    src/tci/parallel.c
    src/tci/slot.c
    src/tci/work_item.c
)

set(TCI_HEADERS
    include/tci/barrier.h
    include/tci/communicator.h
    include/tci/context.h
    include/tci/mutex.h
    include/tci/parallel.h
    include/tci/pipeline.h
    include/tci/slot.h
    include/tci/task_set.h
    include/tci/work_item.h
    include/tci/yield.h
    include/tci.h
    include/tci/barrier.hpp
    include/tci/communicator.hpp
    include/tci/mutex.hpp
    include/tci/parallel.hpp
    include/tci/slot.hpp
    include/tci/work_item.hpp
    include/tci.hpp
)

if(TCI_CXX)
    list(APPEND TCI_SOURCES
        src/tci/communicator.cpp
        src/tci/task_set.cpp
    )
endif()

add_library(tci-objects OBJECT ${TCI_SOURCES})
set_property(TARGET tci-objects PROPERTY POSITION_INDEPENDENT_CODE 1)
target_compile_options(tci-objects PRIVATE $<$<CONFIG:Debug>:-O0>)
target_compile_definitions(tci-objects PUBLIC -D_POSIX_C_SOURCE=200809L)

target_include_directories(tci-objects PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
)

if(TCI_CXX)
    set_target_properties(tci-objects PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
endif()

if(TCI_HAVE_C11_ATOMIC AND NOT TCI_HAVE_GCC_ATOMIC_BUILTINS)
    set_target_properties(tci-objects PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF
    )
endif()

if (ENABLE_SHARED)
    add_library(tci SHARED $<TARGET_OBJECTS:tci-objects>)
    add_library(TCI::tci ALIAS tci)
    target_link_libraries(tci PUBLIC ${PUBLIC_LIBS} PRIVATE ${PRIVATE_LIBS})
    set_target_properties(tci PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
    list(APPEND TCI_TARGETS tci)
    target_include_directories(tci PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>
    )
endif()

if (ENABLE_STATIC)
    add_library(tci-static STATIC $<TARGET_OBJECTS:tci-objects>)
    add_library(TCI::tci-static ALIAS tci-static)
    target_link_libraries(tci-static PUBLIC ${PUBLIC_LIBS} PRIVATE ${PRIVATE_LIBS})
    set_target_properties(tci-static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
    set_target_properties(tci-static PROPERTIES OUTPUT_NAME tci)
    list(APPEND TCI_TARGETS tci-static)
    target_include_directories(tci-static PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>
    )
endif()

list(GET TCI_TARGETS 0 TCI_MAIN_TARGET)

target_sources(${TCI_MAIN_TARGET}
    INTERFACE FILE_SET headers TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include
    FILES ${TCI_HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/include/tci/tci_config.h
)

install(TARGETS ${TCI_TARGETS} EXPORT tci-targets
    FILE_SET headers DESTINATION ${INSTALL_INCLUDEDIR}
)

install(EXPORT tci-targets
    FILE TCITargets.cmake
    NAMESPACE TCI::
    DESTINATION ${INSTALL_LIBDIR}/cmake/TCI
)

write_basic_package_version_file(
    "TCIConfigVersion.cmake"
    VERSION ${CMAKE_PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/TCIConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/TCIConfigVersion.cmake"
    DESTINATION ${INSTALL_LIBDIR}/cmake/TCI
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tci.pc" DESTINATION ${INSTALL_LIBDIR}/pkgconfig)

export(EXPORT tci-targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/TCITargets.cmake
    NAMESPACE TCI::
)

