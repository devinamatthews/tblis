#include "communicator.h"
#include "parallel.h"

#include <stdlib.h>
#include <string.h>

static tci_comm _tci_single = {NULL, 1, 0, 1, 0};
const tci_comm* const tci_single = &_tci_single;

int tci_comm_init_single(tci_comm* comm)
{
    comm->context = NULL;
    comm->nthread = 1;
    comm->tid = 0;
    comm->ngang = 1;
    comm->gid = 0;
    return 0;
}

int tci_comm_init(tci_comm* comm, tci_context* context,
                  unsigned nthread, unsigned tid, unsigned ngang, unsigned gid)
{
    comm->context = context;
    comm->nthread = nthread;
    comm->tid = tid;
    comm->ngang = ngang;
    comm->gid = gid;

    if (context) tci_context_attach(comm->context);

    return 0;
}

int tci_comm_destroy(tci_comm* comm)
{
    if (comm->context)
    {
        return tci_context_detach(comm->context);
    }
    else
    {
        return 0;
    }
}

int tci_comm_is_master(const tci_comm* comm)
{
    return comm->tid == 0;
}

int tci_comm_barrier(tci_comm* comm)
{
    if (!comm->context) return 0;
    return tci_context_barrier(comm->context, comm->tid);
}

int tci_comm_bcast(tci_comm* comm, void** object, unsigned root)
{
    if (!comm->context) return 0;

    if (comm->tid == root)
    {
        return tci_context_send(comm->context, comm->tid, *object);
    }
    else
    {
        return tci_context_receive(comm->context, comm->tid, object);
    }
}

int tci_comm_bcast_nowait(tci_comm* comm, void** object, unsigned root)
{
    if (!comm->context) return 0;

    if (comm->tid == root)
    {
        return tci_context_send_nowait(comm->context, comm->tid, *object);
    }
    else
    {
        return tci_context_receive_nowait(comm->context, comm->tid, object);
    }
}

int tci_comm_gang(tci_comm* parent, tci_comm* child,
                  int type, unsigned n, unsigned bs)
{
    unsigned nt = parent->nthread;
    unsigned tid = parent->tid;

    if (n == 1) return tci_comm_init(child, parent->context, nt, tid, 1, 0);
    if (n >= nt) return tci_comm_init(child, NULL, 1, 0, nt, tid);

    unsigned new_tid = 0;
    unsigned new_nthread = 0;
    unsigned block = 0;

    switch (type & ~TCI_NO_CONTEXT)
    {
        case TCI_EVENLY:
        {
            block = (n*tid)/nt;
            unsigned block_first = (block*nt)/n;
            unsigned block_last = ((block+1)*nt)/n;
            new_tid = tid-block_first;
            new_nthread = block_last-block_first;
        }
        break;
        case TCI_CYCLIC:
        {
            block = tid%n;
            new_tid = tid/n;
            new_nthread = (nt-block+n-1)/n;
        }
        break;
        case TCI_BLOCK_CYCLIC:
        {
            block = (tid/bs)%n;
            unsigned nsubblock_tot = nt/bs;
            unsigned nsubblock = nsubblock_tot/n;
            new_tid = ((tid/bs)/n)*bs + (tid%bs);
            new_nthread = nsubblock*bs +
                TCI_MIN(bs, nt-nsubblock*n*bs-block*bs);
        }
        break;
        case TCI_BLOCKED:
        {
            bs = (nt+n-1)/n;
            block = tid/bs;
            new_tid = tid-block*bs;
            new_nthread = TCI_MIN(bs, nt-block*bs);
        }
        break;
        default: return EINVAL;
    }

    if (!parent->context || (type & TCI_NO_CONTEXT))
    {
        tci_comm_init(child, NULL, new_nthread, new_tid, n, block);
    }
    else
    {
        tci_context* contexts_buf[n];
        tci_context** contexts = &contexts_buf[0];

        memset(contexts_buf, 0, sizeof(contexts_buf));
        tci_comm_bcast_nowait(parent, (void**)&contexts, 0);

        if (new_tid == 0 && new_nthread > 1)
        {
            tci_context_init(&contexts[block], new_nthread,
                             parent->context->barrier.group_size);
        }

        tci_comm_barrier(parent);

        tci_comm_init(child, contexts[block], new_nthread, new_tid, n, block);

        tci_comm_barrier(parent);
    }

    return 0;
}

void tci_distribute(unsigned n, unsigned idx, uint64_t range,
                    uint64_t granularity, uint64_t* first, uint64_t* last,
                    uint64_t* max)
{
    if (n == 1)
    {
        if (first) *first = 0;
        if ( last)  *last = range;
        if (  max)   *max = range;
    }
    else
    {
        uint64_t ngrain = (range+granularity-1)/granularity;
        uint64_t first_grain = (idx*ngrain)/n;
        uint64_t last_grain = ((idx+1)*ngrain)/n;
        uint64_t max_grain = (ngrain+n-1)/n;

        if (first) *first = first_grain*granularity;
        if ( last)  *last = TCI_MIN(last_grain*granularity, range);
        if (  max)   *max = (first_grain+max_grain)*granularity;
    }
}

void tci_comm_distribute_over_gangs(tci_comm* comm, uint64_t range,
                                    uint64_t granularity, uint64_t* first,
                                    uint64_t* last, uint64_t* max)
{
    tci_distribute(comm->ngang, comm->gid, range, granularity,
                   first, last, max);
}

void tci_comm_distribute_over_threads(tci_comm* comm, uint64_t range,
                                      uint64_t granularity, uint64_t* first,
                                      uint64_t* last, uint64_t* max)
{
    tci_distribute(comm->nthread, comm->tid, range, granularity,
                   first, last, max);
}

void tci_comm_distribute_over_gangs_2d(tci_comm* comm,
    uint64_t range_m, uint64_t range_n,
    uint64_t granularity_m, uint64_t granularity_n,
    uint64_t* first_m, uint64_t* last_m, uint64_t* max_m,
    uint64_t* first_n, uint64_t* last_n, uint64_t* max_n)
{
    unsigned m, n;
    tci_partition_2x2(comm->ngang, range_m, range_n, &m, &n);

    unsigned idx_m = comm->gid % m;
    unsigned idx_n = comm->gid / m;

    tci_distribute(m, idx_m, range_m, granularity_m, first_m, last_m, max_m);
    tci_distribute(n, idx_n, range_n, granularity_n, first_n, last_n, max_n);
}

void tci_comm_distribute_over_threads_2d(tci_comm* comm,
     uint64_t range_m, uint64_t range_n,
     uint64_t granularity_m, uint64_t granularity_n,
     uint64_t* first_m, uint64_t* last_m, uint64_t* max_m,
     uint64_t* first_n, uint64_t* last_n, uint64_t* max_n)
{
    unsigned m, n;
    tci_partition_2x2(comm->nthread, range_m, range_n, &m, &n);

    unsigned idx_m = comm->tid % m;
    unsigned idx_n = comm->tid / m;

    tci_distribute(m, idx_m, range_m, granularity_m, first_m, last_m, max_m);
    tci_distribute(n, idx_n, range_n, granularity_n, first_n, last_n, max_n);
}
