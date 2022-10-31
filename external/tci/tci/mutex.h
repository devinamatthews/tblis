#ifndef _TCI_MUTEX_H_
#define _TCI_MUTEX_H_

#include "tci_global.h"

#include "yield.h"

#ifdef __cplusplus
extern "C" {
#endif

#if TCI_USE_OS_UNFAIR_LOICK

typedef os_unfair_lock tci_mutex;

#elif TCI_USE_OSX_SPINLOCK

typedef OSSpinLock tci_mutex;

#elif TCI_USE_PTHREAD_SPINLOCK

typedef pthread_spinlock_t tci_mutex;

#elif TCI_USE_OMP_LOCK

typedef omp_lock_t tci_mutex;

#elif TCI_USE_PTHREAD_MUTEX

typedef pthread_mutex_t tci_mutex;

#else

typedef char tci_mutex;

#endif

int tci_mutex_init(tci_mutex* mutex);

int tci_mutex_destroy(tci_mutex* mutex);

int tci_mutex_lock(tci_mutex* mutex);

int tci_mutexrylock(tci_mutex* mutex);

int tci_mutex_unlock(tci_mutex* mutex);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TCI_DONT_USE_CXX11)

#include <system_error>

namespace tci
{

class mutex
{
    public:
        mutex()
        {
            int ret = tci_mutex_init(&_lock);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }

        ~mutex() noexcept(false)
        {
            int ret = tci_mutex_destroy(&_lock);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }

        mutex(const mutex&) = delete;

        mutex(mutex&&) = default;

        mutex& operator=(const mutex&) = delete;

        mutex& operator=(mutex&&) = default;

        void lock()
        {
            int ret = tci_mutex_lock(&_lock);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }

        bool try_lock()
        {
            int ret = tci_mutexrylock(&_lock);
            if (ret == EBUSY) return false;
            if (ret != 0) throw std::system_error(ret, std::system_category());
            return true;
        }

        void unlock()
        {
            int ret = tci_mutex_unlock(&_lock);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }

        operator tci_mutex*() { return &_lock; }

        operator const tci_mutex*() const { return &_lock; }

    protected:
        tci_mutex _lock;
};

}

#endif

#endif
