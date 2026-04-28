#ifndef _TCI_BARRIER_HPP_
#define _TCI_BARRIER_HPP_

#include "tci/barrier.h"

#include <system_error>

namespace tci
{

class barrier
{
    public:
        barrier(unsigned nthread, unsigned group_size=0)
        {
            int ret = tci_barrier_init(&_barrier, nthread, group_size);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }

        ~barrier() noexcept(false)
        {
            int ret = tci_barrier_destroy(&_barrier);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }

        barrier(const barrier&) = delete;

        barrier(barrier&) = default;

        barrier& operator=(const barrier&) = delete;

        barrier& operator=(barrier&) = default;

        void wait(unsigned tid)
        {
            int ret = tci_barrier_wait(&_barrier, tid);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }

        unsigned num_threads() const
        {
            return _barrier.nthread;
        }

        unsigned group_size() const
        {
            return _barrier.group_size;
        }

        operator tci_barrier*() { return &_barrier; }

        operator const tci_barrier*() const { return &_barrier; }

    protected:
        tci_barrier _barrier;
};

}

#endif
