#ifndef _TCI_COMMUNICATOR_H_
#define _TCI_COMMUNICATOR_H_

#include "tci_config.h"

#include "context.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
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

extern const tci_comm* const tci_single;

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

void tci_distribute(unsigned n, unsigned idx, uint64_t range,
                    uint64_t granularity, uint64_t* first, uint64_t* last,
                    uint64_t* max);

void tci_comm_distribute_over_gangs(tci_comm* comm, uint64_t range,
                                    uint64_t granularity, uint64_t* first,
                                    uint64_t* last, uint64_t* max);

void tci_comm_distribute_over_threads(tci_comm* comm, uint64_t range,
                                      uint64_t granularity, uint64_t* first,
                                      uint64_t* last, uint64_t* max);

void tci_comm_distribute_over_gangs_2d(tci_comm* comm,
    uint64_t range_m, uint64_t range_n,
    uint64_t granularity_m, uint64_t granularity_n,
    uint64_t* first_m, uint64_t* last_m, uint64_t* max_m,
    uint64_t* first_n, uint64_t* last_n, uint64_t* max_n);

void tci_comm_distribute_over_threads_2d(tci_comm* comm,
    uint64_t range_m, uint64_t range_n,
    uint64_t granularity_m, uint64_t granularity_n,
    uint64_t* first_m, uint64_t* last_m, uint64_t* max_m,
    uint64_t* first_n, uint64_t* last_n, uint64_t* max_n);

#ifdef __cplusplus
}

#include <system_error>
#include <tuple>
#include <utility>

namespace tci
{

class communicator
{
    public:
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

        template <typename T>
        void broadcast(T*& object, unsigned root=0) const
        {
            tci_comm_bcast(*this, reinterpret_cast<void**>(&object), root);
        }

        template <typename T>
        void broadcast_nowait(T*& object, unsigned root=0) const
        {
            tci_comm_bcast_nowait(*this, reinterpret_cast<void**>(&object), root);
        }

        communicator gang(int type, unsigned n, unsigned bs=0) const
        {
            communicator child;
            int ret = tci_comm_gang(*this, &child._comm, type, n, bs);
            if (ret != 0) throw std::system_error(ret, std::system_category());
            return child;
        }

        std::tuple<uint64_t,uint64_t,uint64_t>
        distribute_over_gangs(uint64_t range, uint64_t granularity=1) const
        {
            uint64_t first, last, max;
            tci_comm_distribute_over_gangs(*this, range, granularity,
                                           &first, &last, &max);
            return std::make_tuple(first, last, max);
        }

        std::tuple<uint64_t,uint64_t,uint64_t>
        distribute_over_threads(uint64_t range, uint64_t granularity=1) const
        {
            uint64_t first, last, max;
            tci_comm_distribute_over_threads(*this, range, granularity,
                                             &first, &last, &max);
            return std::make_tuple(first, last, max);
        }

        std::tuple<uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t>
        distribute_over_gangs_2d(uint64_t range_m, uint64_t range_n,
                                 uint64_t granularity_m=1,
                                 uint64_t granularity_n=1) const
        {
            uint64_t first_m, last_m, max_m, first_n, last_n, max_n;
            tci_comm_distribute_over_gangs_2d(*this, range_m, range_n,
                                              granularity_m, granularity_n,
                                              &first_m, &last_m, &max_m,
                                              &first_n, &last_n, &max_n);
            return std::make_tuple(first_m, last_m, max_m,
                                   first_n, last_n, max_n);
        }

        std::tuple<uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t>
        distribute_over_threads_2d(uint64_t range_m, uint64_t range_n,
                                   uint64_t granularity_m=1,
                                   uint64_t granularity_n=1) const
        {
            uint64_t first_m, last_m, max_m, first_n, last_n, max_n;
            tci_comm_distribute_over_threads_2d(*this, range_m, range_n,
                                                granularity_m, granularity_n,
                                                &first_m, &last_m, &max_m,
                                                &first_n, &last_n, &max_n);
            return std::make_tuple(first_m, last_m, max_m,
                                   first_n, last_n, max_n);
        }
        
        operator tci_comm*() const { return const_cast<tci_comm*>(&_comm); }

    protected:
        tci_comm _comm;
};

}

#endif

#endif
