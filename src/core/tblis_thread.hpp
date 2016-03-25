#ifndef _TBLIS_THREAD_HPP_
#define _TBLIS_THREAD_HPP_

#define USE_SPIN_BARRIER 1

#include <condition_variable>

#include "tblis_mutex.hpp"

#include "util/util.hpp"

#ifndef BLIS_TREE_BARRIER_ARITY
#define BLIS_TREE_BARRIER_ARITY 0
#endif

#if USE_OPENMP
#include <omp.h>
#elif USE_PTHREADS
#include <pthread.h>
#else
#include <thread>
#endif

namespace tblis
{

namespace detail
{

#if USE_PTHREAD_BARRIER && !BLIS_OS_OSX

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

#elif USE_SPIN_BARRIER

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
                while (step.load(std::memory_order_acquire) == old_step) {}
            }
        }
    };

#else

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

        std::vector<int> thread_barrier;
        bool is_tree = true;

        TreeBarrier(int nthread, int group_size=BLIS_TREE_BARRIER_ARITY)
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
                nleaders = (nleaders+group_size-1)/group_size;
                nbarrier += nleaders;
            }
            while (nleaders > 1);

            barriers = (Barrier*)bli_malloc(sizeof(Barrier)*nbarrier);

            int idx = 0;
            int nchildren = nthread;
            do
            {
                int nparents = (nchildren+group_size-1)/group_size;
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

            thread_barrier.resize(nthread);
            for (int i = 0;i < nthread;i++)
                thread_barrier[i] = i/group_size;
        }

        TreeBarrier(const TreeBarrier&) = delete;

        TreeBarrier& operator=(const TreeBarrier&) = delete;

        ~TreeBarrier()
        {
            if (is_tree)
            {
                if (barriers) bli_free(barriers);
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
                barriers[thread_barrier[tid]].wait();
            }
            else
            {
                barrier.wait();
            }
        }
    };

}

class ThreadContext
{
    friend class ThreadCommunicator;
    template <typename Body> friend void parallelize(int nthread, Body&& body);

    public:
        void barrier(int tid)
        {
            _barrier.wait(tid);
        }

        int num_threads() const
        {
            return _nthread;
        }

        void send(int tid, void* object)
        {
            _buffer = object;
            barrier(tid);
            barrier(tid);
        }

        void* receive(int tid)
        {
            barrier(tid);
            void* object = _buffer;
            barrier(tid);
            return object;
        }

    protected:
        ThreadContext(int nthread)
        : _barrier(nthread), _nthread(nthread) {}

        detail::TreeBarrier _barrier;
        void* _buffer = NULL;
        int _nthread;
};

#if USE_PTHREADS
namespace detail
{
    template <typename Body> void* run_thread(void* raw_data);
}
#endif

class ThreadCommunicator
{
    template <typename Body> friend void parallelize(int nthread, Body&& body);
#if USE_PTHREADS
    template <typename Body> friend void* detail::run_thread(void* raw_data);
#endif

    public:
        ThreadCommunicator()
        : _context(), _nthread(1), _tid(0), _gid(0) {}

        ThreadCommunicator(const ThreadCommunicator&) = delete;

        ThreadCommunicator(ThreadCommunicator&&) = default;

        ThreadCommunicator& operator=(const ThreadCommunicator&) = delete;

        ThreadCommunicator& operator=(ThreadCommunicator&&) = default;

        void barrier()
        {
            if (_nthread == 1) return;
            _context->barrier(_tid);
        }

        int num_threads() const
        {
            return _nthread;
        }

        int thread_num() const
        {
            return _tid;
        }

        int gang_num() const
        {
            return _gid;
        }

        template <typename T>
        void broadcast(T*& object, int root=0)
        {
            if (_nthread == 1) return;

            if (_tid == root)
            {
                _context->send(_tid, object);
            }
            else
            {
                object = static_cast<T*>(_context->receive(_tid));
            }
        }

        ThreadCommunicator gang_evenly(int n)
        {
            int block = (n*_tid)/_nthread;
            int block_first = (block*_nthread)/n;
            int block_last = ((block+1)*_nthread)/n;
            int new_tid = _tid-block_first;
            int new_nthread = block_last-block_first;

            return gang(n, block, new_tid, new_nthread);
        }

        ThreadCommunicator gang_block_cyclic(int n, int bs)
        {
            int block = (_tid/bs)%n;
            int nsubblock_tot = _nthread/bs;
            int nsubblock = nsubblock_tot/n;
            int new_tid = ((_tid/bs)/n)*bs + (_tid%bs);
            int new_nthread = nsubblock*bs + std::min(bs, _nthread-nsubblock*n*bs-block*bs);

            return gang(n, block, new_tid, new_nthread);
        }

        ThreadCommunicator gang_blocked(int n)
        {
            int bs = (_nthread+n-1)/n;
            int block = _tid/bs;
            int new_tid = _tid-block*bs;
            int new_nthread = std::min(bs, _nthread-block*bs);

            return gang(n, block, new_tid, new_nthread);
        }

        ThreadCommunicator gang_cyclic(int n)
        {
            int block = _tid%n;
            int new_tid = _tid/n;
            int new_nthread = (_nthread-block+n-1)/n;

            return gang(n, block, new_tid, new_nthread);
        }

        std::pair<dim_t,dim_t> distribute_over_gangs(int ngang, dim_t n, dim_t granularity=1)
        {
            return distribute(ngang, _gid, n, granularity);
        }

        std::pair<dim_t,dim_t> distribute_over_threads(dim_t n, dim_t granularity=1)
        {
            return distribute(_nthread, _tid, n, granularity);
        }

    protected:
        ThreadCommunicator(const std::shared_ptr<ThreadContext>& context, int tid, int gid)
        : _context(context), _nthread(context->num_threads()), _tid(tid), _gid(gid) {}

        ThreadCommunicator gang(int n, int block, int new_tid, int new_nthread)
        {
            if (new_nthread == 1) return ThreadCommunicator();

            std::shared_ptr<ThreadContext>* contexts;
            std::vector<std::shared_ptr<ThreadContext>> contexts_root;
            if (_tid == 0)
            {
                contexts_root.resize(n);
                contexts = contexts_root.data();
            }
            broadcast(contexts);

            if (new_tid == 0)
            {
                contexts[block].reset(new ThreadContext(new_nthread));
            }
            barrier();

            ThreadCommunicator new_comm(contexts[block], new_tid, block);
            barrier();

            return new_comm;
        }

        std::pair<dim_t,dim_t> distribute(int nelem, int elem, dim_t n, dim_t granularity)
        {
            dim_t ng = (n+granularity-1)/granularity;
            return {         (( elem   *ng)/nelem)*granularity,
                    std::min((((elem+1)*ng)/nelem)*granularity, n)};
        }

        std::shared_ptr<ThreadContext> _context;
        int _nthread;
        int _tid;
        int _gid;
};

#if USE_OPENMP

template <typename Body>
void parallelize(int nthread, Body&& body)
{
    if (nthread > 1)
    {
        std::shared_ptr<ThreadContext> context(new ThreadContext(nthread));

        #pragma omp parallel num_threads(nthread)
        {
            ThreadCommunicator comm(context, omp_get_thread_num(), 0);
            Body body_copy(body);
            body_copy(comm);
        }
    }
    else
    {
        ThreadCommunicator comm;
        body(comm);
    }
}

#elif USE_PTHREADS

namespace detail
{
    template <typename Body>
    struct thread_data
    {
        const Body& body;
        const std::shared_ptr<ThreadContext>& context;
        int tid;

        thread_data(const Body& body,
                    const std::shared_ptr<ThreadContext>& context,
                    int tid)
        : body(body), context(context), tid(tid) {}
    };

    template <typename Body>
    void* run_thread(void* raw_data)
    {
        thread_data<Body>& data = *static_cast<thread_data<Body>*>(raw_data);
        ThreadCommunicator comm(data.context, data.tid, 0);
        Body body_copy(data.body);
        body_copy(comm);
        return NULL;
    }
}

template <typename Body>
void parallelize(int nthread, Body&& body)
{
    std::vector<pthread_t> threads; threads.reserve(nthread);
    std::vector<detail::thread_data<Body>> data; data.reserve(nthread);

    ThreadCommunicator comm;
    std::shared_ptr<ThreadContext> context;
    if (nthread > 1)
    {
        context.reset(new ThreadContext(nthread));
        comm = ThreadCommunicator(context, 0, 0);
    }

    for (int i = 1;i < nthread;i++)
    {
        threads.emplace_back();
        data.emplace_back(body, context, i);
        int err = pthread_create(&threads.back(), NULL,
                                 detail::run_thread<Body>, &data.back());
        if (err != 0) throw std::system_error(err, std::generic_category());

    }

    body(comm);

    for (auto& t : threads)
    {
        int err = pthread_join(t, NULL);
        if (err != 0) throw std::system_error(err, std::generic_category());
    }
}

#else

template <typename Body>
void parallelize(int nthread, Body&& body)
{
    std::vector<std::thread> threads; threads.reserve(nthread);

    ThreadCommunicator comm;
    std::shared_ptr<ThreadContext> context;
    if (nthread > 1)
    {
        context.reset(new ThreadContext(nthread));
        comm = ThreadCommunicator(context, 0, 0);
    }

    for (int i = 1;i < nthread;i++)
    {
        threads.emplace_back(
        [=,&context]() mutable
        {
            ThreadCommunicator comm(context, i, 0);
            body(comm);
        });
    }

    body(comm);

    for (auto& t : threads)
    {
        std::cout << t.joinable() << std::endl;
        t.join();
    }
}

#endif

}

#endif
