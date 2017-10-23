#ifndef _TBLIS_THREAD_HPP_
#define _TBLIS_THREAD_HPP_

#ifdef TBLIS_DONT_USE_CXX11
#ifndef TCI_DONT_USE_CXX11
#define TCI_DONT_USE_CXX11 1
#endif
#endif

#include "tci.h"
#include "basic_types.h"

#if defined(__cplusplus) && TBLIS_ENABLE_TBB
#define TBLIS_SIMPLE_TBB 1
#include <tbb/tbb.h>
#endif

typedef tci_comm tblis_comm;
extern const tblis_comm* const tblis_single;

#ifdef __cplusplus
extern "C"
{
#endif

unsigned tblis_get_num_threads();

void tblis_set_num_threads(unsigned num_threads);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

#include <vector>
#include <utility>
#include <atomic>

namespace tblis
{

template <typename T>
void atomic_accumulate(std::atomic<T>& x, T y)
{
    T old = x.load();
    while (!x.compare_exchange_weak(old, old+y)) continue;
}

using namespace tci;

extern tci::communicator single;

inline int max_num_threads(const communicator& comm)
{
    (void)comm;
    #if TBLIS_ENABLE_TBB
    #if TBB_INTERFACE_VERSION >= 9100
        return tbb::this_task_arena::max_concurrency();
    #else //TBB_INTERFACE_VERSION >= 9100
        return tblis_get_num_threads();
    #endif //TBB_INTERFACE_VERSION >= 9100
    #else //TBLIS_ENABLE_TBB
        return comm.num_threads();
    #endif //TBLIS_ENABLE_TBB
}

#if TBLIS_ENABLE_TBB

class dynamic_task_set
{
    public:
        dynamic_task_set(const communicator&, int, len_type) {}

        ~dynamic_task_set()
        {
            group_.wait();
        }

        template <typename Func>
        void visit(int task, Func&& f)
        {
            group_.run([=] { const_cast<Func&>(f)(single); });
        }

    protected:
        tbb::task_group group_;
        static len_type inout_ratio;
};

#else //TBLIS_ENABLE_TBB

class dynamic_task_set
{
    public:
        dynamic_task_set(const communicator& comm, int ntask, len_type nwork)
        : comm_(comm), ntask_(ntask)
        {
            if (comm_.master()) slots_ = new slot<>[ntask];
            comm_.broadcast_value(slots_);

            int nt = max_num_threads(comm_);
            int nt_outer, nt_inner;
            std::tie(nt_outer, nt_inner) =
                partition_2x2(nt, inout_ratio*nwork, ntask,
                              nwork, nt);

            subcomm_ = comm_.gang(TCI_EVENLY, nt_outer);
        }

        ~dynamic_task_set()
        {
            comm_.barrier();
            if (comm_.master()) delete[] slots_;
        }

        template <typename Func>
        void visit(int task, Func&& f)
        {
            TBLIS_ASSERT(task >= 0 && task < ntask_);
            if (slots_[task].try_fill(subcomm_.gang_num())) f(subcomm_);
        }

    protected:
        const communicator& comm_;
        communicator subcomm_;
        slot<>* slots_ = nullptr;
        int ntask_;
        static len_type inout_ratio;
};

#endif //TBLIS_ENABLE_TBB

template <typename T>
void reduce_init(reduce_t op, T& value, len_type& idx)
{
    typedef std::numeric_limits<real_type_t<T>> limits;

    switch (op)
    {
        case REDUCE_SUM:
        case REDUCE_SUM_ABS:
        case REDUCE_MAX_ABS:
        case REDUCE_NORM_2:
            value = T();
            break;
        case REDUCE_MAX:
            value = limits::lowest();
            break;
        case REDUCE_MIN:
        case REDUCE_MIN_ABS:
            value = limits::max();
            break;
    }

    idx = -1;
}

template <typename T>
void reduce(const communicator& comm, reduce_t op, T& value, len_type& idx)
{
    if (comm.num_threads() == 1)
    {
        if (op == REDUCE_NORM_2) value = sqrt(value);
        return;
    }

    std::vector<std::pair<T,len_type>> vals;
    if (comm.master()) vals.resize(comm.num_threads());

    comm.broadcast(
    [&](std::vector<std::pair<T,len_type>>& vals)
    {
        vals[comm.thread_num()] = {value, idx};
    },
    vals);

    if (comm.master())
    {
        if (op == REDUCE_SUM)
        {
            for (unsigned i = 1;i < comm.num_threads();i++)
            {
                vals[0].first += vals[i].first;
            }
        }
        else if (op == REDUCE_SUM_ABS)
        {
            vals[0].first = std::abs(vals[0].first);
            for (unsigned i = 1;i < comm.num_threads();i++)
            {
                vals[0].first += std::abs(vals[i].first);
            }
        }
        else if (op == REDUCE_MAX)
        {
            for (unsigned i = 1;i < comm.num_threads();i++)
            {
                if (vals[i].first > vals[0].first) vals[0] = vals[i];
            }
        }
        else if (op == REDUCE_MAX_ABS)
        {
            for (unsigned i = 1;i < comm.num_threads();i++)
            {
                if (std::abs(vals[i].first) >
                    std::abs(vals[0].first)) vals[0] = vals[i];
            }
        }
        else if (op == REDUCE_MIN)
        {
            for (unsigned i = 1;i < comm.num_threads();i++)
            {
                if (vals[i].first < vals[0].first) vals[0] = vals[i];
            }
        }
        else if (op == REDUCE_MIN_ABS)
        {
            for (unsigned i = 1;i < comm.num_threads();i++)
            {
                if (std::abs(vals[i].first) <
                    std::abs(vals[0].first)) vals[0] = vals[i];
            }
        }
        else if (op == REDUCE_NORM_2)
        {
            for (unsigned i = 1;i < comm.num_threads();i++)
            {
                vals[0].first += vals[i].first;
            }
            vals[0].first = std::sqrt(vals[0].first);
        }

        value = vals[0].first;
        idx = vals[0].second;
    }

    comm.barrier();
}

template <typename Func, typename... Args>
void parallelize_if(const Func& f, const tblis_comm* _comm, Args&&... args)
{
    if (_comm)
    {
        f(*reinterpret_cast<const communicator*>(_comm), args...);
    }
    else
    {
        parallelize
        (
            [&,f](const communicator& comm) mutable
            {
                f(comm, args...);
                comm.barrier();
            },
            tblis_get_num_threads()
        );
    }
}

}

#endif

#endif
