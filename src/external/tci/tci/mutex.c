#include "mutex.h"

#include <string.h>

#if TCI_USE_OSX_SPINLOCK

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
    *mutex = 0;
    return 0;
}

int tci_mutex_destroy(tci_mutex* mutex)
{
    (void)mutex;
    return 0;
}

int tci_mutex_lock(tci_mutex* mutex)
{
    while (__atomic_test_and_set(mutex, __ATOMIC_ACQUIRE)) tci_yield();
    return 0;
}

int tci_mutex_trylock(tci_mutex* mutex)
{
    if (!__atomic_test_and_set(mutex, __ATOMIC_ACQUIRE))
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
    __atomic_clear((bool*)mutex, __ATOMIC_RELEASE);
    return 0;
}

#endif
