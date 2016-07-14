#ifndef _TBLIS_BARRIER_HPP_
#define _TBLIS_BARRIER_HPP_

#include "tblis.hpp"

#if TBLIS_USE_PTHREAD_BARRIER
#include <cerrno>
#include <system_error>
#include <pthread.h>
#elif TBLIS_USE_SPIN_BARRIER
#include <atomic>
#elif TBLIS_USE_CXX11_BARRIER
#include <mutex>
#include <condition_variable>
#endif

namespace tblis
{

#if TBLIS_USE_PTHREAD_BARRIER

    struct Barrier
    {
        Barrier* parent;
        pthread_barrier_t barrier;

        Barrier(Barrier* parent, int nchildren)
        : parent(parent)
        {
            if (pthread_barrier_init(&barrier, NULL, nchildren) != 0)
                throw std::system_error("Unable to init barrier");
        }

        ~Barrier()
        {
            if (pthread_barrier_destroy(&barrier) != 0)
                throw std::system_error("Unable to destroy barrier");
        }

        void wait()
        {
            int ret = pthread_barrier_wait(&barrier);
            if (ret != 0 && ret != PTHREAD_BARRIER_SERIAL_THREAD)
                throw std::system_error("Unable to wait on barrier");
            if (parent)
            {
                if (ret == PTHREAD_BARRIER_SERIAL_THREAD)
                    parent->wait();

                ret = pthread_barrier_wait(&barrier);
                if (ret != 0 && ret != PTHREAD_BARRIER_SERIAL_THREAD)
                    throw std::system_error("Unable to wait on barrier");
            }
        }
    };

#elif TBLIS_USE_SPIN_BARRIER

    struct Barrier
    {
        Barrier* parent;
        int nchildren;
        std::atomic<unsigned> step;
        std::atomic<int> nwaiting;

        Barrier(Barrier* parent, int nchildren)
        : parent(parent), nchildren(nchildren), step(0), nwaiting(0) {}

        void wait()
        {
            unsigned old_step = step.load(std::memory_order_acquire);

            if (nwaiting.fetch_add(1, std::memory_order_acq_rel) == nchildren-1)
            {
                if (parent) parent->wait();
                nwaiting.store(0, std::memory_order_release);
                step.fetch_add(1, std::memory_order_acq_rel);
            }
            else
            {
                while (step.load(std::memory_order_acquire) == old_step) yield();
            }
        }
    };

#elif TBLIS_USE_CXX11_BARRIER

    struct Barrier
    {
        Barrier* parent;
        std::mutex lock;
        std::condition_variable condvar;
        unsigned step = 0;
        int nchildren;
        volatile int nwaiting;

        Barrier(Barrier* parent, int nchildren)
        : parent(parent), nchildren(nchildren), nwaiting(0) {}

        void wait()
        {
            std::unique_lock<std::mutex> guard(lock);

            unsigned old_step = step;

            if (++nwaiting == nchildren)
            {
                if (parent) parent->wait();
                nwaiting = 0;
                step++;
                guard.unlock();
                condvar.notify_all();
            }
            else
            {
                while (step == old_step) condvar.wait(guard);
            }
        }
    };

#endif

struct TreeBarrier
{
    union
    {
        Barrier* barriers;
        Barrier barrier;
    };

    int nthread;
    int group_size;
    bool is_tree = true;

    TreeBarrier(int nthread, int group_size=0)
    : nthread(nthread), group_size(group_size)
    {
        if (group_size == 0 || group_size >= nthread)
        {
            is_tree = false;
            new (&barrier) Barrier(NULL, nthread);
            return;
        }

        int nbarrier = 0;
        int nleaders = nthread;
        do
        {
            nleaders = ceil_div(nleaders, group_size);
            nbarrier += nleaders;
        }
        while (nleaders > 1);

        barriers = (Barrier*)::operator new(sizeof(Barrier)*nbarrier);

        int idx = 0;
        int nchildren = nthread;
        do
        {
            int nparents = ceil_div(nchildren, group_size);
            for (int i = 0;i < nparents;i++)
            {
                new (barriers+idx+i)
                    Barrier(barriers+idx+nparents+i/group_size,
                            std::min(group_size, nchildren-i*group_size));
            }
            idx += nparents;
            nchildren = nparents;
        }
        while (nchildren > 1);
    }

    TreeBarrier(const TreeBarrier&) = delete;

    TreeBarrier& operator=(const TreeBarrier&) = delete;

    ~TreeBarrier()
    {
        if (is_tree)
        {
            int nbarrier = 0;
            int nleaders = nthread;
            do
            {
                nleaders = ceil_div(nleaders, group_size);
                nbarrier += nleaders;
            }
            while (nleaders > 1);

            for (int i = 0;i < nbarrier;i++) barriers[i].~Barrier();
            ::operator delete(barriers);
        }
        else
        {
            barrier.~Barrier();
        }
    }

    void wait(int tid)
    {
        if (is_tree)
        {
            barriers[tid/group_size].wait();
        }
        else
        {
            barrier.wait();
        }
    }

    int arity() const
    {
        return group_size;
    }
};

}

#endif
