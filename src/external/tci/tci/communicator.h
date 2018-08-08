#ifndef _TCI_COMMUNICATOR_H_
#define _TCI_COMMUNICATOR_H_

#include "tci_global.h"

#include "context.h"

typedef struct tci_comm
{
    tci_context* context;
    unsigned ngang;
    unsigned gid;
    unsigned nthread;
    unsigned tid;
} tci_comm;

enum
{
    TCI_EVENLY         = (1<<1),
    TCI_CYCLIC         = (2<<1),
    TCI_BLOCK_CYCLIC   = (3<<1),
    TCI_BLOCKED        = (4<<1),
    TCI_NO_CONTEXT     =    0x1
};

typedef struct tci_range
{
    uint64_t size;
    uint64_t grain;

#ifdef __cplusplus
    tci_range() : size(0), grain(1) {}

    template <typename T>
    tci_range(const T& size) : size(size), grain(1) {}

    template <typename T, typename U>
    tci_range(const T& size, const U& grain) : size(size), grain(grain) {}
#endif
} tci_range;

typedef void (*tci_range_func)(tci_comm*, uint64_t, uint64_t, void*);

typedef void (*tci_range_2d_func)(tci_comm*, uint64_t, uint64_t,
                                  uint64_t, uint64_t, void*);

#ifdef __cplusplus
extern "C" {
#endif

extern tci_comm* const tci_single;

int tci_comm_init_single(tci_comm* comm);

int tci_comm_init(tci_comm* comm, tci_context* context,
                  unsigned nthread, unsigned tid, unsigned ngang, unsigned gid);

int tci_comm_destroy(tci_comm* comm);

int tci_comm_is_master(const tci_comm* comm);

int tci_comm_barrier(tci_comm* comm);

int tci_comm_bcast(tci_comm* comm, void** object, unsigned root);

int tci_comm_bcast_nowait(tci_comm* comm, void** object, unsigned root);

int tci_comm_gang(tci_comm* parent, tci_comm* child,
                  int type, unsigned n, unsigned bs);

void tci_comm_distribute_over_gangs(tci_comm* comm, tci_range range,
                                    tci_range_func func, void* payload);

void tci_comm_distribute_over_threads(tci_comm* comm, tci_range range,
                                      tci_range_func func, void* payload);

void tci_comm_distribute_over_gangs_2d(tci_comm* comm, tci_range range_m,
                                       tci_range range_n,
                                       tci_range_2d_func func, void* payload);

void tci_comm_distribute_over_threads_2d(tci_comm* comm, tci_range range_m,
                                         tci_range range_n,
                                         tci_range_2d_func func, void* payload);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TCI_DONT_USE_CXX11)

#include <system_error>
#include <tuple>
#include <utility>

namespace tci
{
    class communicator;
}

#include "task_set.h"

namespace tci
{

namespace detail
{

#if __cplusplus >= 201402l

using std::index_sequence;
using std::index_sequence_for;

#else

template <size_t... S>
struct index_sequence
{
    typedef size_t value_type;
    static constexpr size_t size() noexcept { return sizeof...(S); }
};

template <typename, typename> struct concat_sequences;

template <size_t... S, size_t... R>
struct concat_sequences<index_sequence<S...>, index_sequence<R...>>
{
    typedef index_sequence<S..., (R+sizeof...(S))...> type;
};

template <size_t N, typename=void> struct make_index_sequence_helper;

template <size_t N> struct make_index_sequence_helper<N, typename std::enable_if<N==0>::type>
{
    typedef index_sequence<> type;
};

template <size_t N> struct make_index_sequence_helper<N, typename std::enable_if<N==1>::type>
{
    typedef index_sequence<0> type;
};

template <size_t N> struct make_index_sequence_helper<N, typename std::enable_if<(N>1)>::type>
{
    typedef typename concat_sequences<typename make_index_sequence_helper<(N+1)/2>::type,
                                      typename make_index_sequence_helper<N/2>::type>::type type;
};

template <size_t N>
using make_index_sequence = typename make_index_sequence_helper<N>::type;

template <typename... T>
using index_sequence_for = make_index_sequence<sizeof...(T)>;

#endif

}

class communicator
{
    protected:
        template <typename Func, typename... Args>
        struct broadcast_from_internal
        {
            template <size_t... I>
            broadcast_from_internal(const communicator& comm, unsigned root,
                                    Func&& func, detail::index_sequence<I...>,
                                    Args&&... args)
            {
                std::tuple<Args&&...> refs(std::forward<Args>(args)...);
                auto ptr = &refs;
                tci_comm_bcast(comm, reinterpret_cast<void**>(&ptr), root);
                func(std::get<I>(*ptr)...);
                comm.barrier();
            }
        };

    public:
        class deferred_task_set
        {
            friend class communicator;

            public:
                ~deferred_task_set()
                {
                    tci_task_set_destroy(&_tasks);
                }

                template <typename Func>
                void visit(unsigned task, Func&& func)
                {
                    typedef typename std::decay<Func>::type RealFunc;
                    RealFunc* payload = new RealFunc(std::forward<Func>(func));
                    tci_task_set_visit(&_tasks,
                    [](tci_comm* comm, unsigned, void* payload_)
                    {
                        RealFunc* payload = (RealFunc*)payload_;
                        (*payload)(*reinterpret_cast<const communicator*>(comm));
                        delete payload;
                    }, task, payload);
                }

            protected:
                deferred_task_set(const communicator& comm,
                                  unsigned ntask, int64_t work)
                {
                    tci_task_set_init(&_tasks, comm, ntask, work);
                }

                template <typename Func>
                void visit_all(Func&& func)
                {
                    tci_task_set_visit_all(&_tasks,
                    [](tci_comm* comm, unsigned task, void* payload)
                    {
                        (*(typename std::decay<Func>::type*)payload)
                            (*reinterpret_cast<const communicator*>(comm), task);
                    }, &func);
                }

                tci_task_set _tasks;
        };

        communicator()
        {
            tci_comm_init_single(*this);
        }

        ~communicator()
        {
            tci_comm_destroy(*this);
        }

        communicator(const communicator&) = delete;

        communicator(communicator&& other)
        : _comm(other._comm)
        {
            other._comm.context = nullptr;
        }

        communicator& operator=(const communicator&) = delete;

        communicator& operator=(communicator&& other)
        {
            std::swap(_comm, other._comm);
            return *this;
        }

        bool master() const
        {
            return tci_comm_is_master(*this);
        }

        void barrier() const
        {
            int ret = tci_comm_barrier(*this);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }

        unsigned num_threads() const
        {
            return _comm.nthread;
        }

        unsigned thread_num() const
        {
            return _comm.tid;
        }

        unsigned num_gangs() const
        {
            return _comm.ngang;
        }

        unsigned gang_num() const
        {
            return _comm.gid;
        }

        template <typename Func, typename... Args>
        void broadcast_from(unsigned root, Func&& func, Args&&... args) const
        {
            broadcast_from_internal<Func, Args...>
                (*this, root, std::forward<Func>(func),
                 detail::index_sequence_for<Args...>{},
                 std::forward<Args>(args)...);
        }

        template <typename Func, typename... Args>
        void broadcast(Func&& func, Args&&... args) const
        {
            broadcast_from(0, std::forward<Func>(func),
                           std::forward<Args>(args)...);
        }

        template <typename Arg>
        void broadcast_value_from(unsigned root, Arg& arg) const
        {
            unsigned tid = thread_num();
            broadcast_from(root,
            [&](Arg& master)
            {
                if (tid != root) arg = master;
            },
            arg);
        }

        template <typename Arg>
        void broadcast_value(Arg& arg) const
        {
            broadcast_value_from(0, arg);
        }

        communicator gang(int type, unsigned n, unsigned bs=0) const
        {
            communicator child;
            int ret = tci_comm_gang(*this, &child._comm, type, n, bs);
            if (ret != 0) throw std::system_error(ret, std::system_category());
            return child;
        }

        template <typename Func>
        void distribute_over_gangs(const tci_range& n, Func&& func) const
        {
            tci_comm_distribute_over_gangs(*this, n,
            [](tci_comm*, uint64_t first, uint64_t last, void* payload)
            {
                (*(typename std::decay<Func>::type*)payload)(first, last);
            }, &func);
        }

        template <typename Func>
        void distribute_over_threads(const tci_range& n, Func&& func) const
        {
            tci_comm_distribute_over_threads(*this, n,
            [](tci_comm*, uint64_t first, uint64_t last, void* payload)
            {
                (*(typename std::decay<Func>::type*)payload)(first, last);
            }, &func);
        }

        template <typename Func>
        void distribute_over_gangs(const tci_range& m, const tci_range& n,
                                   Func&& func) const
        {
            tci_comm_distribute_over_gangs_2d(*this, m, n,
            [](tci_comm*, uint64_t mfirst, uint64_t mlast,
               uint64_t nfirst, uint64_t nlast, void* payload)
            {
                (*(typename std::decay<Func>::type*)payload)
                    (mfirst, mlast, nfirst, nlast);
            }, &func);
        }

        template <typename Func>
        void distribute_over_threads(const tci_range& m, const tci_range& n,
                                     Func&& func) const
        {
            tci_comm_distribute_over_threads_2d(*this, m, n,
            [](tci_comm*, uint64_t mfirst, uint64_t mlast,
               uint64_t nfirst, uint64_t nlast, void* payload)
            {
                (*(typename std::decay<Func>::type*)payload)
                    (mfirst, mlast, nfirst, nlast);
            }, &func);
        }

        template <typename Func>
        void iterate_over_gangs(unsigned n, Func&& func) const
        {
            tci_comm_distribute_over_gangs(*this, n,
            [](tci_comm*, uint64_t first, uint64_t last, void* payload)
            {
                for (unsigned i = first;i < last;i++)
                {
                    (*(typename std::decay<Func>::type*)payload)(i);
                }
            }, &func);
        }

        template <typename Func>
        void iterate_over_threads(unsigned n, Func&& func) const
        {
            tci_comm_distribute_over_threads(*this, n,
            [](tci_comm*, uint64_t first, uint64_t last, void* payload)
            {
                for (unsigned i = first;i < last;i++)
                {
                    (*(typename std::decay<Func>::type*)payload)(i);
                }
            }, &func);
        }

        template <typename Func>
        void iterate_over_gangs(unsigned m, unsigned n, Func&& func) const
        {
            typedef std::pair<typename std::decay<Func>::type,unsigned> payload_type;

            payload_type payload{&func, n};

            tci_comm_distribute_over_gangs(*this, m*n,
            [](tci_comm*, uint64_t first, uint64_t last, void* payload)
            {
                unsigned n = ((payload_type*)payload)->second;
                unsigned m0 = first / n;
                unsigned n0 = first % n;
                unsigned m1 = last / n;
                unsigned n1 = last % n;

                for (unsigned i = m0;i <= m1;i++)
                {
                    for (unsigned j = (i == m0 ? n0 : 0);
                                  j < (i == m1 ? n1 : n);j++)
                    {
                        ((payload_type*)payload)->first(i, j);
                    }
                }
            }, &payload);
        }

        template <typename Func>
        void iterate_over_threads(unsigned m, unsigned n, Func&& func) const
        {
            typedef std::pair<typename std::decay<Func>::type,unsigned> payload_type;

            payload_type payload{&func, n};

            tci_comm_distribute_over_threads(*this, m*n,
            [](tci_comm*, uint64_t first, uint64_t last, void* payload)
            {
                unsigned n = ((payload_type*)payload)->second;
                unsigned m0 = first / n;
                unsigned n0 = first % n;
                unsigned m1 = last / n;
                unsigned n1 = last % n;

                for (unsigned i = m0;i <= m1;i++)
                {
                    for (unsigned j = (i == m0 ? n0 : 0);
                                  j < (i == m1 ? n1 : n);j++)
                    {
                        ((payload_type*)payload)->first(i, j);
                    }
                }
            }, &payload);
        }

        template <typename Func>
        void do_tasks_deferred(unsigned ntask, int64_t work, Func&& func) const
        {
            deferred_task_set tasks(*this, ntask, work);
            func(tasks);
        }

        template <typename Func>
        void do_tasks_deferred(unsigned ntask, Func&& func) const
        {
            do_tasks_deferred(ntask, 0, func);
        }

        template <typename Func>
        void do_tasks(unsigned ntask, int64_t work, Func&& func) const
        {
            deferred_task_set tasks(*this, ntask, work);
            tasks.visit_all(func);
        }

        template <typename Func>
        void do_tasks(unsigned ntask, Func&& func) const
        {
            do_tasks(ntask, 0, func);
        }

        operator tci_comm*() const { return const_cast<tci_comm*>(&_comm); }

    protected:
        tci_comm _comm;
};

}

#endif

#endif
