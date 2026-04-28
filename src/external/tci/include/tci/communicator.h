#ifndef _TCI_COMMUNICATOR_H_
#define _TCI_COMMUNICATOR_H_

#include "tci/tci_config.h"
#include "tci/context.h"

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

void tci_distribute(unsigned n, unsigned idx, tci_comm* comm,
                    tci_range range, tci_range_func func, void* payload);

void tci_distribute_2d(unsigned num, unsigned idx, tci_comm* comm,
                       tci_range range_m, tci_range range_n,
                       tci_range_2d_func func, void* payload);

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

#endif
