#ifndef _TCI_MUTEX_H_
#define _TCI_MUTEX_H_

#include "tci/tci_config.h"

#if TCI_USE_OS_UNFAIR_LOICK

#include <os/lock.h>

typedef os_unfair_lock tci_mutex;

#elif TCI_USE_OSX_SPINLOCK

#include <libkern/OSAtomic.h>

typedef OSSpinLock tci_mutex;

#elif TCI_USE_PTHREAD_SPINLOCK

#include <pthread.h>

typedef pthread_spinlock_t tci_mutex;

#elif TCI_USE_OMP_LOCK

#include <omp.h>

typedef omp_lock_t tci_mutex;

#elif TCI_USE_PTHREAD_MUTEX

typedef pthread_mutex_t tci_mutex;

#else

typedef TCI_ATOMIC_FLAG tci_mutex;

#endif

#ifdef __cplusplus
extern "C" {
#endif

int tci_mutex_init(tci_mutex* mutex);

int tci_mutex_destroy(tci_mutex* mutex);

int tci_mutex_lock(tci_mutex* mutex);

int tci_mutexrylock(tci_mutex* mutex);

int tci_mutex_unlock(tci_mutex* mutex);

#ifdef __cplusplus
}
#endif

#endif
