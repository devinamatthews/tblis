#include "mutex.h"

#include <stdarg.h>
#include <stdio.h>
#include <errno.h>

#define TBLIS_USE_OMP_LOCK 1

#if TBLIS_USE_OSX_SPINLOCK
#include <libkern/OSAtomic.h>
#elif TBLIS_USE_PTHREAD_SPINLOCK || TBLIS_USE_PTHREAD_MUTEX
#include <pthread.h>
#elif TBLIS_USE_C11_ATOMIC_SPINLOCK
#include <stdatomic.h>
#elif TBLIS_USE_GNU_ATOMIC_SPINLOCK
#include <stdbool.h>
#elif TBLIS_USE_OMP_LOCK
#include <omp.h>
#endif

#if TBLIS_USE_OSX_SPINLOCK

int tci_mutex_init(tci_mutex_t* mutex)
{
    static OSSpinLock init = OS_SPINLOCK_INIT;
    memcpy((OSSpinLock*)mutex, &init, sizeof(OSSpinLock));
    return 0;
}

int tci_mutex_destroy(tci_mutex_t* mutex)
{
    return 0;
}

int tci_mutex_lock(tci_mutex_t* mutex)
{
    OSSpinLockLock((OSSpinLock*)mutex);
    return 0;
}

int tci_mutex_trylock(tci_mutex_t* mutex)
{
    if (OSSpinLockTry((OSSpinLock*)mutex))
    {
        return 0;
    }
    else
    {
        return EBUSY;
    }
}

int tci_mutex_unlock(tci_mutex_t* mutex)
{
    OSSpinLockUnlock((OSSpinLock*)mutex);
    return 0;
}

#elif TBLIS_USE_PTHREAD_SPINLOCK

int tci_mutex_init(tci_mutex_t* mutex)
{
    return pthread_spin_init((pthread_spinlock_t*)mutex,
                             PTHREAD_PROCESS_PRIVATE);
}

int tci_mutex_destroy(tci_mutex_t* mutex)
{
    return pthread_spin_destroy((pthread_spinlock_t*)mutex);
}

int tci_mutex_lock(tci_mutex_t* mutex)
{
    return pthread_spin_lock((pthread_spinlock_t*)mutex);
}

int tci_mutex_trylock(tci_mutex_t* mutex)
{
    return pthread_spin_trylock((pthread_spinlock_t*)mutex);
}

int tci_mutex_unlock(tci_mutex_t* mutex)
{
    return pthread_spin_unlock((pthread_spinlock_t*)mutex);
}

class mutex
{
    public:
        mutex()
        {
            if (pthread_spin_init(&_lock, PTHREAD_PROCESS_PRIVATE) != 0)
                throw std::system_error("Unable to init spinlock");
        }

        ~mutex()
        {
            if (pthread_spin_destroy(&_lock) != 0)
                throw std::system_error("Unable to destroy spinlock");
        }

        mutex(const mutex&) = delete;

        mutex& operator=(const mutex&) = delete;

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

#elif TBLIS_USE_C11_ATOMIC_SPINLOCK

int tci_mutex_init(tci_mutex_t* mutex)

int tci_mutex_init(tci_mutex_t* mutex)
{
    struct atomic_flag init = ATOMIC_FLAG_INIT;
    memcpy((struct atomic_flag*)mutex, &init, sizeof(struct atomic_flag));
    return 0;
}

int tci_mutex_destroy(tci_mutex_t* mutex)
{
    return 0;
}

int tci_mutex_lock(tci_mutex_t* mutex)
{
    while (atomic_flag_test_and_set((struct atomic_flag*)mutex))
        tci_yield();
    return 0;
}

int tci_mutex_trylock(tci_mutex_t* mutex)
{
    if (!atomic_flag_test_and_set((struct atomic_flag*)mutex))
    {
        return 0;
    }
    else
    {
        return EBUSY;
    }
}

int tci_mutex_unlock(tci_mutex_t* mutex)
{
    atomic_flag_clear((struct atomic_flag*)mutex);
    return 0;
}

#elif TBLIS_USE_GNU_ATOMIC_SPINLOCK

int tci_mutex_init(tci_mutex_t* mutex)
{
    *(bool*)mutex = 0;
    return 0;
}

int tci_mutex_destroy(tci_mutex_t* mutex)
{
    return 0;
}

int tci_mutex_lock(tci_mutex_t* mutex)
{
    while (__atomic_test_and_set((bool*)mutex, __ATOMIC_SEQ_CST))
        tci_yield();
    return 0;
}

int tci_mutex_trylock(tci_mutex_t* mutex)
{
    if (!__atomic_test_and_set((bool*)mutex, __ATOMIC_SEQ_CST))
    {
        return 0;
    }
    else
    {
        return EBUSY;
    }
}

int tci_mutex_unlock(tci_mutex_t* mutex)
{
    __atomic_clear((bool*)mutex, __ATOMIC_SEQ_CST);
    return 0;
}

#elif TBLIS_USE_GNU_SYNC_SPINLOCK

int tci_mutex_init(tci_mutex_t* mutex)
{
    *(int*)mutex = 0;
    return 0;
}

int tci_mutex_destroy(tci_mutex_t* mutex)
{
    return 0;
}

int tci_mutex_lock(tci_mutex_t* mutex)
{
    while (__sync_lock_test_and_set((int*)mutex, 1))
        tci_yield();
    return 0;
}

int tci_mutex_trylock(tci_mutex_t* mutex)
{
    if (!__sync_lock_test_and_set((int*)mutex, 1))
    {
        return 0;
    }
    else
    {
        return EBUSY;
    }
}

int tci_mutex_unlock(tci_mutex_t* mutex)
{
    __sync_lock_release((int*)mutex);
    return 0;
}

#elif TBLIS_USE_PTHREAD_MUTEX

int tci_mutex_init(tci_mutex_t* mutex)
{
    return pthread_mutex_init((pthread_mutex_t*)mutex, NULL);
}

int tci_mutex_destroy(tci_mutex_t* mutex)
{
    return pthread_mutex_destroy((pthread_mutex_t*)mutex);
}

int tci_mutex_lock(tci_mutex_t* mutex)
{
    return pthread_mutex_lock((pthread_mutex_t*)mutex);
}

int tci_mutex_trylock(tci_mutex_t* mutex)
{
    return pthread_mutex_trylock((pthread_mutex_t*)mutex);
}

int tci_mutex_unlock(tci_mutex_t* mutex)
{
    return pthread_mutex_unlock((pthread_mutex_t*)mutex);
}

#elif TBLIS_USE_OMP_LOCK

int tci_mutex_init(tci_mutex_t* mutex)
{
    omp_init_lock((omp_lock_t*)mutex);
    return 0;
}

int tci_mutex_destroy(tci_mutex_t* mutex)
{
    omp_destroy_lock((omp_lock_t*)mutex);
    return 0;
}

int tci_mutex_lock(tci_mutex_t* mutex)
{
    omp_set_lock((omp_lock_t*)mutex);
    return 0;
}

int tci_mutex_trylock(tci_mutex_t* mutex)
{
    if (omp_test_lock((omp_lock_t*)mutex))
    {
        return 0;
    }
    else
    {
        return EBUSY;
    }
}

int tci_mutex_unlock(tci_mutex_t* mutex)
{
    omp_unset_lock((omp_lock_t*)mutex);
    return 0;
}

#endif

mutex print_lock;

void printf_locked(const char* fmt, ...)
{
    std::lock_guard<mutex> guard(print_lock);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

}
