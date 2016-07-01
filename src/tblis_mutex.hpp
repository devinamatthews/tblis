#ifndef _TBLIS_MUTEX_HPP_
#define _TBLIS_MUTEX_HPP_

#include "tblis.hpp"

#if TBLIS_USE_OSX_SPINLOCK
#include <libkern/OSAtomic.h>
#elif TBLIS_USE_PTHREAD_SPINLOCK || TBLIS_USE_PTHREAD_MUTEX
#include <pthread.h>
#elif TBLIS_USE_CXX11_SPINLOCK
#include <atomic>
#elif TBLIS_USE_OMP_LOCK
#include <omp.h>
#endif

namespace tblis
{

#if TBLIS_USE_OSX_SPINLOCK

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

#elif TBLIS_USE_PTHREAD_SPINLOCK

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

#elif TBLIS_USE_CXX11_SPINLOCK

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

#elif TBLIS_USE_PTHREAD_MUTEX

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

#elif TBLIS_USE_OMP_LOCK

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

#elif TBLIS_USE_CXX11_MUTEX

using Mutex = std::mutex;

#endif

extern void printf_locked(const char* fmt, ...);

}

#endif
