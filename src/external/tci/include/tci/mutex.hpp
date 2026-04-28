#ifndef _TCI_MUTEX_HPP_
#define _TCI_MUTEX_HPP_

#include "tci/mutex.h"

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
