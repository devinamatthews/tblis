#include "tci/mutex.h"
#include "tci/yield.h"

#include <string.h>

#if TCI_USE_OS_UNFAIR_LOCK

#include <os/lock.h>

int tci_mutex_init(tci_mutex* mutex)
{
    static os_unfair_lock init = OS_UNFAIR_LOCK_INIT;
    memcpy(mutex, &init, sizeof(os_unfair_lock));
    return 0;
}

int tci_mutex_destroy(tci_mutex* mutex)
{
    (void)mutex;
    return 0;
}

int tci_mutex_lock(tci_mutex* mutex)
{
    os_unfair_lock_lock(mutex);
    return 0;
}

int tci_mutex_trylock(tci_mutex* mutex)
{
    if (os_unfair_lock_trylock(mutex))
    {
        return 0;
    }
    else
    {
        return EBUSY;
    }
}

int tci_mutex_unlock(tci_mutex* mutex)
{
    os_unfair_lock_unlock(mutex);
    return 0;
}

#elif TCI_USE_OSX_SPINLOCK

#include <libkern/OSAtomic.h>

int tci_mutex_init(tci_mutex* mutex)
{
    static OSSpinLock init = OS_SPINLOCK_INIT;
    memcpy(mutex, &init, sizeof(OSSpinLock));
    return 0;
}

int tci_mutex_destroy(tci_mutex* mutex)
{
    (void)mutex;
    return 0;
}

int tci_mutex_lock(tci_mutex* mutex)
{
    OSSpinLockLock(mutex);
    return 0;
}

int tci_mutex_trylock(tci_mutex* mutex)
{
    if (OSSpinLockTry(mutex))
    {
        return 0;
    }
    else
    {
        return EBUSY;
    }
}

int tci_mutex_unlock(tci_mutex* mutex)
{
    OSSpinLockUnlock(mutex);
    return 0;
}

#elif TCI_USE_PTHREAD_SPINLOCK

#include <pthread.h>

int tci_mutex_init(tci_mutex* mutex)
{
    return pthread_spin_init(mutex, PTHREAD_PROCESS_PRIVATE);
}

int tci_mutex_destroy(tci_mutex* mutex)
{
    return pthread_spin_destroy(mutex);
}

int tci_mutex_lock(tci_mutex* mutex)
{
    return pthread_spin_lock(mutex);
}

int tci_mutex_trylock(tci_mutex* mutex)
{
    return pthread_spin_trylock(mutex);
}

int tci_mutex_unlock(tci_mutex* mutex)
{
    return pthread_spin_unlock(mutex);
}

#elif TCI_USE_PTHREAD_MUTEX

#include <pthread.h>

int tci_mutex_init(tci_mutex* mutex)
{
    return pthread_mutex_init(mutex, NULL);
}

int tci_mutex_destroy(tci_mutex* mutex)
{
    return pthread_mutex_destroy(mutex);
}

int tci_mutex_lock(tci_mutex* mutex)
{
    return pthread_mutex_lock(mutex);
}

int tci_mutex_trylock(tci_mutex* mutex)
{
    return pthread_mutex_trylock(mutex);
}

int tci_mutex_unlock(tci_mutex* mutex)
{
    return pthread_mutex_unlock(mutex);
}

#elif TCI_USE_OMP_LOCK

#include <omp.h>

int tci_mutex_init(tci_mutex* mutex)
{
    omp_init_lock(mutex);
    return 0;
}

int tci_mutex_destroy(tci_mutex* mutex)
{
    omp_destroy_lock(mutex);
    return 0;
}

int tci_mutex_lock(tci_mutex* mutex)
{
    omp_set_lock(mutex);
    return 0;
}

int tci_mutex_trylock(tci_mutex* mutex)
{
    if (omp_test_lock(mutex))
    {
        return 0;
    }
    else
    {
        return EBUSY;
    }
}

int tci_mutex_unlock(tci_mutex* mutex)
{
    omp_unset_lock(mutex);
    return 0;
}

#else

int tci_mutex_init(tci_mutex* mutex)
{
    tci_atomic_clear(mutex, TCI_ATOMIC_RELAXED);
    return 0;
}

int tci_mutex_destroy(tci_mutex* mutex)
{
    (void)mutex;
    return 0;
}

int tci_mutex_lock(tci_mutex* mutex)
{
    while (tci_atomic_test_and_set(mutex, TCI_ATOMIC_ACQUIRE)) tci_yield();
    return 0;
}

int tci_mutex_trylock(tci_mutex* mutex)
{
    if (!tci_atomic_test_and_set(mutex, TCI_ATOMIC_ACQUIRE))
    {
        return 0;
    }
    else
    {
        return EBUSY;
    }
}

int tci_mutex_unlock(tci_mutex* mutex)
{
    tci_atomic_clear(mutex, TCI_ATOMIC_RELEASE);
    return 0;
}

#endif
