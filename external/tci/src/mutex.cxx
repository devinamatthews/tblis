#include "mutex.h"
#include "yield.h"

#include <cerrno>
#include <system_error>

#if TBLIS_USE_CXX11_ATOMIC_SPINLOCK
#include <atomic>
#elif TBLIS_USE_CXX11_MUTEX
#include <mutex>
#endif

extern "C"
{

#if TBLIS_USE_CXX11_ATOMIC_SPINLOCK

int tci_mutex_init(tci_mutex_t* mutex)
{
    std::atomic_flag& _flag = *(std::atomic_flag*)mutex;
    new (&_flag) std::atomic_flag(ATOMIC_FLAG_INIT);
    return 0;
}

int tci_mutex_destroy(tci_mutex_t* mutex)
{
    std::atomic_flag& _flag = *(std::atomic_flag*)mutex;
    _flag.~atomic_flag();
    return 0;
}

int tci_mutex_lock(tci_mutex_t* mutex)
{
    std::atomic_flag& _flag = *(std::atomic_flag*)mutex;
    while (_flag.test_and_set()) tci_yield();
    return 0;
}

int tci_mutex_trylock(tci_mutex_t* mutex)
{
    std::atomic_flag& _flag = *(std::atomic_flag*)mutex;
    if (!_flag.test_and_set())
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
    std::atomic_flag& _flag = *(std::atomic_flag*)mutex;
    _flag.clear();
    return 0;
}

#elif TBLIS_USE_CXX11_MUTEX

int tci_mutex_init(tci_mutex_t* mutex)
{
    std::mutex& _mutex = *(std::mutex*)mutex;
    new (&_mutex) std::mutex();
    return 0;
}

int tci_mutex_destroy(tci_mutex_t* mutex)
{
    std::mutex& _mutex = *(std::mutex*)mutex;
    _mutex.~mutex();
    return 0;
}

int tci_mutex_lock(tci_mutex_t* mutex)
{
    std::mutex& _mutex = *(std::mutex*)mutex;
    try
    {
        _mutex.lock();
    }
    catch (std::system_error& e)
    {
        return e.code().value();
    }
    return 0;
}

int tci_mutex_trylock(tci_mutex_t* mutex)
{
    std::mutex& _mutex = *(std::mutex*)mutex;
    if (_mutex.try_lock())
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
    std::mutex& _mutex = *(std::mutex*)mutex;
    _mutex.unlock();
    return 0;
}

#endif

}
