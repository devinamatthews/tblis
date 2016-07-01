#ifndef _TBLIS_THREAD_HPP_
#define _TBLIS_THREAD_HPP_

#include "tblis.hpp"

#if TBLIS_USE_OPENMP_THREADS
#include <omp.h>
#elif TBLIS_USE_PTHREAD_THREADS
#include <pthread.h>
#elif TBLIS_USE_CXX11_THREADS
#include <thread>
#endif

namespace tblis
{

namespace detail
{
    template <typename Body>
    void parallelize(Body& body, int nthread, int arity);

    template <typename Body> void* run_thread(void* raw_data);
}

class ThreadContext
{
    friend class ThreadCommunicator;
    template <typename Body> friend void detail::parallelize(Body& body, int nthread, int arity);

    public:
        ThreadContext(int nthread, int arity=0)
        : _barrier(nthread, arity), _nthread(nthread) {}

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

        void send_nowait(int tid, void* object)
        {
            _buffer = object;
            barrier(tid);
        }

        void* receive(int tid)
        {
            barrier(tid);
            void* object = _buffer;
            barrier(tid);
            return object;
        }

        void* receive_nowait(int tid)
        {
            barrier(tid);
            void* object = _buffer;
            return object;
        }

    protected:
        TreeBarrier _barrier;
        void* _buffer = NULL;
        int _nthread;
};

class ThreadCommunicator
{
    template <typename Body> friend void detail::parallelize(Body& body, int nthread, int arity);
    template <typename Body> friend void* detail::run_thread(void* raw_data);

    public:
        ThreadCommunicator()
        : _context(), _nthread(1), _tid(0), _gid(0) {}

        ThreadCommunicator(const ThreadCommunicator&) = delete;

        ThreadCommunicator(ThreadCommunicator&&) = default;

        ThreadCommunicator& operator=(const ThreadCommunicator&) = delete;

        ThreadCommunicator& operator=(ThreadCommunicator&&) = default;

        bool master() const
        {
            return _tid == 0;
        }

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

        template <typename T>
        void broadcast_nowait(T*& object, int root=0)
        {
            if (_nthread == 1) return;

            if (_tid == root)
            {
                _context->send_nowait(_tid, object);
            }
            else
            {
                object = static_cast<T*>(_context->receive_nowait(_tid));
            }
        }

        ThreadCommunicator gang_evenly(int n)
        {
            if (n >= _nthread) return ThreadCommunicator(_tid);

            int block = (n*_tid)/_nthread;
            int block_first = (block*_nthread)/n;
            int block_last = ((block+1)*_nthread)/n;
            int new_tid = _tid-block_first;
            int new_nthread = block_last-block_first;

            return gang(n, block, new_tid, new_nthread);
        }

        ThreadCommunicator gang_block_cyclic(int n, int bs)
        {
            if (n >= _nthread) return ThreadCommunicator(_tid);

            int block = (_tid/bs)%n;
            int nsubblock_tot = _nthread/bs;
            int nsubblock = nsubblock_tot/n;
            int new_tid = ((_tid/bs)/n)*bs + (_tid%bs);
            int new_nthread = nsubblock*bs + std::min(bs, _nthread-nsubblock*n*bs-block*bs);

            return gang(n, block, new_tid, new_nthread);
        }

        ThreadCommunicator gang_blocked(int n)
        {
            if (n >= _nthread) return ThreadCommunicator(_tid);

            int bs = (_nthread+n-1)/n;
            int block = _tid/bs;
            int new_tid = _tid-block*bs;
            int new_nthread = std::min(bs, _nthread-block*bs);

            return gang(n, block, new_tid, new_nthread);
        }

        ThreadCommunicator gang_cyclic(int n)
        {
            if (n >= _nthread) return ThreadCommunicator(_tid);

            int block = _tid%n;
            int new_tid = _tid/n;
            int new_nthread = (_nthread-block+n-1)/n;

            return gang(n, block, new_tid, new_nthread);
        }

        std::tuple<idx_type,idx_type,idx_type> distribute_over_gangs(int ngang, idx_type n, idx_type granularity=1)
        {
            return distribute(ngang, _gid, n, granularity);
        }

        std::tuple<idx_type,idx_type,idx_type> distribute_over_threads(idx_type n, idx_type granularity=1)
        {
            return distribute(_nthread, _tid, n, granularity);
        }

    protected:
        ThreadCommunicator(int gid)
        : _nthread(1), _tid(0), _gid(gid) {}

        ThreadCommunicator(const std::shared_ptr<ThreadContext>& context, int tid, int gid)
        : _context(context), _nthread(context->num_threads()), _tid(tid), _gid(gid) {}

        ThreadCommunicator gang(int n, int block, int new_tid, int new_nthread)
        {
            ThreadCommunicator new_comm;

            std::shared_ptr<ThreadContext>* contexts;
            std::vector<std::shared_ptr<ThreadContext>> contexts_root;
            if (master())
            {
                contexts_root.resize(n);
                contexts = contexts_root.data();
            }
            broadcast_nowait(contexts);

            if (new_tid == 0 && new_nthread > 1)
            {
                contexts[block] = std::make_shared<ThreadContext>(new_nthread, _context->_barrier.arity());
            }

            barrier();

            if (new_nthread > 1)
                new_comm = ThreadCommunicator(contexts[block], new_tid, block);

            barrier();

            return new_comm;
        }

        std::tuple<idx_type,idx_type,idx_type> distribute(int nelem, int elem, idx_type n, idx_type granularity)
        {
            idx_type ng = (n+granularity-1)/granularity;
            idx_type max_size = ((ng+nelem-1)/nelem)*granularity;

            return std::tuple<idx_type,idx_type,idx_type>
                (         (( elem   *ng)/nelem)*granularity,
                 std::min((((elem+1)*ng)/nelem)*granularity, n),
                          (( elem   *ng)/nelem)*granularity+max_size);
        }

        std::shared_ptr<ThreadContext> _context;
        int _nthread;
        int _tid;
        int _gid;
};

namespace detail
{

#if TBLIS_USE_OPENMP_THREADS

template <typename Body>
void parallelize(Body& body, int nthread, int arity)
{
    std::shared_ptr<ThreadContext> context = std::make_shared<ThreadContext>(nthread, arity);

    #pragma omp parallel num_threads(nthread)
    {
        ThreadCommunicator comm(context, omp_get_thread_num(), 0);
        Body body_copy(body);
        body_copy(comm);
    }
}

#elif TBLIS_USE_PTHREAD_THREADS

template <typename Body>
struct thread_data
{
    Body& body;
    const std::shared_ptr<ThreadContext>& context;
    int tid;

    thread_data(Body& body,
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

template <typename Body>
void parallelize(Body& body, int nthread, int arity)
{
    std::vector<pthread_t> threads; threads.reserve(nthread);
    std::vector<detail::thread_data<Body>> data; data.reserve(nthread);

    std::shared_ptr<ThreadContext> context = std::make_shared<ThreadContext>(nthread, arity);
    ThreadCommunicator comm(context, 0, 0);

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

#elif TBLIS_USE_CXX11_THREADS

template <typename Body>
void parallelize(Body& body, int nthread, int arity)
{
    std::vector<std::thread> threads; threads.reserve(nthread);

    std::shared_ptr<ThreadContext> context = std::make_shared<ThreadContext>(nthread, arity);
    ThreadCommunicator comm(context, 0, 0);

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

    for (auto& t : threads) t.join();
}

#endif

}

template <typename Body>
void parallelize(Body body, int nthread, int arity=0)
{
    if (nthread > 1)
    {
        detail::parallelize(body, nthread, arity);
    }
    else
    {
        ThreadCommunicator comm;
        body(comm);
    }
}

}

#endif
