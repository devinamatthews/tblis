#ifndef _TBLIS_MUTEX_HPP_
#define _TBLIS_MUTEX_HPP_

#define USE_OPENMP 0
#define USE_PTHREADS 1

#include <atomic>
#include <mutex>

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

#if USE_OSSPINLOCK && BLIS_OS_OSX
#include <libkern/OSAtomic.h>
#endif

#if USE_PTHREADS
#include <pthread.h>
#include <cerrno>
#include <system_error>
#elif USE_OPENMP
#include <omp.h>
#else
#include <thread>
#endif

#if TBLIS_ARCH_INTEL
#include <xmmintrin.h>
#endif

namespace tblis
{

#if TBLIS_ARCH_MIC

inline void yield()
{
    _mm_delay(32);
}

#elif TBLIS_ARCH_INTEL

inline void yield()
{
    //_mm_pause();
    __asm__ __volatile__ ("pause");
}

#else

inline void yield() {}

#endif

#if USE_SPINLOCK

#if USE_OSSPINLOCK && BLIS_OS_OSX

class Mutex
{
    public:
        Mutex(const Mutex&) = delete;

        Mutex& operator=(const Mutex&) = delete;

        void lock()
        {
            OSSpinLockLock(&_lock);
        }

        bool try_lock()
        {
            return OSSpinLockTry(&_lock);
        }

        void unlock()
        {
            OSSpinLockUnlock(&_lock);
        }

    protected:
        OSSpinLock _lock = OS_SPINLOCK_INIT;
};

#elif USE_PTHREADS && !BLIS_OS_OSX

class Mutex
{
    public:
        Mutex()
        {
            if (thread_spin_init(&_lock, PTHREAD_PROCESS_PRIVATE) != 0)
                throw std::system_error("Unable to init spinlock");
        }

        ~Mutex()
        {
            if (pthread_spin_destroy(&_lock) != 0)
                throw std::system_error("Unable to destroy spinlock");
        }

        Mutex(const Mutex&) = delete;

        Mutex& operator=(const Mutex&) = delete;

        void lock()
        {
            if (pthread_spin_lock(&_lock) != 0)
                throw std::system_error("Unable to lock spinlock");
        }

        bool try_lock()
        {
            int ret = pthread_spin_trylock(&_lock);
            if (ret == 0) return true;
            if (ret == EBUSY) return false;
            throw std::system_error("Unable to lock spinlock");
        }

        void unlock()
        {
            if (pthread_spin_unlock(&_lock) != 0)
                throw std::system_error("Unable to unlock spinlock");
        }

    protected:
        pthread_spinlock_t _lock;
};

#else

class Mutex
{
    public:
        constexpr Mutex() noexcept {}

        Mutex(const Mutex&) = delete;

        Mutex& operator=(const Mutex&) = delete;

        void lock()
        {
            while (_flag.test_and_set(std::memory_order_acquire)) yield();
        }

        bool try_lock()
        {
            return !_flag.test_and_set(std::memory_order_acquire);
        }

        void unlock()
        {
            _flag.clear(std::memory_order_release);
        }

    protected:
        std::atomic_flag _flag = ATOMIC_FLAG_INIT;
};

#endif

#else

#if USE_PTHREADS

class Mutex
{
    public:
        constexpr Mutex() noexcept {}

        Mutex(const Mutex&) = delete;

        Mutex& operator=(const Mutex&) = delete;

        void lock()
        {
            int err = pthread_mutex_lock(&_mutex);
            if (err != 0) throw std::system_error(err, std::generic_category());
        }

        bool try_lock()
        {
            int err = pthread_mutex_trylock(&_mutex);
            if (err == 0) return true;
            if (err != EBUSY) throw std::system_error(err, std::generic_category());
            return false;
        }

        void unlock()
        {
            int err = pthread_mutex_unlock(&_mutex);
            if (err != 0) throw std::system_error(err, std::generic_category());
        }

    protected:
        pthread_mutex_t _mutex = PTHREAD_MUTEX_INITIALIZER;
};

#elif USE_OPENMP

class Mutex
{
    public:
        Mutex()
        {
            omp_init_lock(&_mutex);
        }

        ~Mutex()
        {
            omp_destroy_lock(&_mutex);
        }

        Mutex(const Mutex&) = delete;

        Mutex& operator=(const Mutex&) = delete;

        void lock()
        {
            omp_set_lock(&_mutex);
        }

        bool try_lock()
        {
            return omp_test_lock(&_mutex);
        }

        void unlock()
        {
            omp_unset_lock(&_mutex);
        }

    protected:
        omp_lock_t _mutex;
};

#else

using Mutex = std::mutex;

#endif

#endif

extern void printf_locked(const char* fmt, ...);

}

#endif
