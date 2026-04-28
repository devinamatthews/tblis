#include "tci/communicator.h"
#include "tci/parallel.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

static tci_comm _tci_single = {NULL, 1, 0, 1, 0};
tci_comm* const tci_single = &_tci_single;

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
            unsigned block_first = (block*nt+n-1)/n;
            unsigned block_last = ((block+1)*nt+n-1)/n;
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

#if TCI_USE_OPENMP_THREADS || TCI_USE_PTHREADS_THREADS || TCI_USE_WINDOWS_THREADS

void tci_distribute(unsigned n, unsigned idx, tci_comm* comm,
                    tci_range range, tci_range_func func, void* payload)
{
    if (n == 1)
    {
        func(comm, 0, range.size, payload);
        return;
    }

    range.grain = TCI_MAX(range.grain, 1);

    uint64_t ngrain = (range.size+range.grain-1)/range.grain;
    uint64_t first = (idx*ngrain)/n;
    uint64_t last = ((idx+1)*ngrain)/n;

    func(comm, first*range.grain, TCI_MIN(last*range.grain, range.size), payload);
}

void tci_distribute_2d(unsigned num, unsigned idx, tci_comm* comm,
                       tci_range range_m, tci_range range_n,
                       tci_range_2d_func func, void* payload)
{
    if (num == 1)
    {
        func(comm, 0, range_m.size, 0, range_n.size, payload);
        return;
    }

    unsigned m, n;
    tci_partition_2x2(num, range_m.size, num, range_n.size, num, &m, &n);

    unsigned idx_m = idx % m;
    unsigned idx_n = idx / m;

    range_m.grain = TCI_MAX(range_m.grain, 1);
    range_n.grain = TCI_MAX(range_n.grain, 1);

    uint64_t mgrain = (range_m.size+range_m.grain-1)/range_m.grain;
    uint64_t ngrain = (range_n.size+range_n.grain-1)/range_n.grain;
    uint64_t mfirst = (idx_m*mgrain)/m;
    uint64_t nfirst = (idx_n*ngrain)/n;
    uint64_t mlast = ((idx_m+1)*mgrain)/m;
    uint64_t nlast = ((idx_n+1)*ngrain)/n;

    func(comm, mfirst*range_m.grain, TCI_MIN(mlast*range_m.grain, range_m.size),
               nfirst*range_n.grain, TCI_MIN(nlast*range_n.grain, range_n.size), payload);
}

#elif TCI_USE_OMPTASK_THREADS

void tci_distribute(unsigned n, unsigned idx, tci_comm* comm,
                    tci_range range, tci_range_func func, void* payload)
{
    (void)idx;

    if (n == 1)
    {
        func(comm, 0, range.size, payload);
        return;
    }

    range.grain = TCI_MAX(range.grain, 1);
    uint64_t ngrain = (range.size+range.grain-1)/range.grain;

    #pragma omp taskgroup
    {
        for (uint64_t idx = 0;idx < n;idx++)
        {
            #pragma omp task
            {
                uint64_t first = (idx*ngrain)/n;
                uint64_t last = ((idx+1)*ngrain)/n;

                func(comm, first*range.grain, TCI_MIN(last*range.grain, range.size), payload);
            }
        }
    }
}

void tci_distribute_2d(unsigned num, unsigned idx, tci_comm* comm,
                       tci_range range_m, tci_range range_n,
                       tci_range_2d_func func, void* payload)
{
    (void)idx;

    if (num == 1)
    {
        func(comm, 0, range_m.size, 0, range_n.size, payload);
        return;
    }

    unsigned m, n;
    tci_partition_2x2(num, range_m.size, num, range_n.size, num, &m, &n);

    range_m.grain = TCI_MAX(range_m.grain, 1);
    range_n.grain = TCI_MAX(range_n.grain, 1);
    uint64_t mgrain = (range_m.size+range_m.grain-1)/range_m.grain;
    uint64_t ngrain = (range_n.size+range_n.grain-1)/range_n.grain;

    #pragma omp taskgroup
    {
        for (uint64_t idx_m = 0;idx_m < m;idx_m++)
        {
            for (uint64_t idx_n = 0;idx_n < n;idx_n++)
            {
                #pragma omp task
                {
                    uint64_t mfirst = (idx_m*mgrain)/m;
                    uint64_t nfirst = (idx_n*ngrain)/n;
                    uint64_t mlast = ((idx_m+1)*mgrain)/m;
                    uint64_t nlast = ((idx_n+1)*ngrain)/n;

                    func(comm, mfirst*range_m.grain, TCI_MIN(mlast*range_m.grain, range_m.size),
                               nfirst*range_n.grain, TCI_MIN(nlast*range_n.grain, range_n.size), payload);
                }
            }
        }
    }
}

#elif TCI_USE_DISPATCH_THREADS

#include <dispatch/dispatch.h>

typedef struct tci_distribute_func_data
{
    tci_comm* comm;
    tci_range_func func;
    uint64_t n;
    tci_range* range;
    void* payload;
} tci_distribute_func_data;

static void tci_distribute_func(void* data_, size_t idx)
{
    tci_distribute_func_data* data = (tci_distribute_func_data*)data_;

    uint64_t ngrain = (data->range->size+data->range->grain-1)/data->range->grain;
    uint64_t first = (idx*ngrain)/data->n;
    uint64_t last = ((idx+1)*ngrain)/data->n;

    data->func(data->comm, first*data->range->grain,
               TCI_MIN(last*data->range->grain, data->range->size), data->payload);
}

typedef struct tci_distribute_2d_func_data
{
    tci_comm* comm;
    tci_range_2d_func func;
    uint64_t m, n;
    tci_range *range_m, *range_n;
    void* payload;
} tci_distribute_2d_func_data;

static void tci_distribute_2d_func(void* data_, size_t idx)
{
    tci_distribute_2d_func_data* data = (tci_distribute_2d_func_data*)data_;

    unsigned idx_m = idx % data->m;
    unsigned idx_n = idx / data->m;

    uint64_t mgrain = (data->range_m->size+data->range_m->grain-1)/data->range_m->grain;
    uint64_t ngrain = (data->range_n->size+data->range_n->grain-1)/data->range_n->grain;
    uint64_t mfirst = (idx_m*mgrain)/data->m;
    uint64_t nfirst = (idx_n*ngrain)/data->n;
    uint64_t mlast = ((idx_m+1)*mgrain)/data->m;
    uint64_t nlast = ((idx_n+1)*ngrain)/data->n;

    data->func(data->comm, mfirst*data->range_m->grain,
               TCI_MIN(mlast*data->range_m->grain, data->range_m->size),
               nfirst*data->range_n->grain,
               TCI_MIN(nlast*data->range_n->grain, data->range_n->size), data->payload);
}

void tci_distribute(unsigned n, unsigned idx, tci_comm* comm,
                    tci_range range, tci_range_func func, void* payload)
{
    (void)idx;

    if (n == 1)
    {
        func(comm, 0, range.size, payload);
        return;
    }

    range.grain = TCI_MAX(range.grain, 1);

    tci_distribute_func_data data = {comm, func, n, &range, payload};
    dispatch_queue_t queue =
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

    dispatch_apply_f(n, queue, &data, tci_distribute_func);
}

void tci_distribute_2d(unsigned num, unsigned idx, tci_comm* comm,
                       tci_range range_m, tci_range range_n,
                       tci_range_2d_func func, void* payload)
{
    (void)idx;

    if (num == 1)
    {
        func(comm, 0, range_m.size, 0, range_n.size, payload);
        return;
    }

    unsigned m, n;
    tci_partition_2x2(num, range_m.size, num, range_n.size, num, &m, &n);

    range_m.grain = TCI_MAX(range_m.grain, 1);
    range_n.grain = TCI_MAX(range_n.grain, 1);

    tci_distribute_2d_func_data data = {comm, func, m, n, &range_m, &range_n, payload};
    dispatch_queue_t queue =
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

    dispatch_apply_f(m*n, queue, &data, tci_distribute_2d_func);
}

#elif TCI_USE_TBB_THREADS || TCI_USE_PPL_THREADS

// defined in communicator.cpp

#else // single threaded

void tci_distribute(unsigned n, unsigned idx, tci_comm* comm,
                    tci_range range, tci_range_func func, void* payload)
{
    (void)n;
    (void)idx;
    func(comm, 0, range.size, payload);
}

void tci_distribute_2d(unsigned n, unsigned idx, tci_comm* comm,
                       tci_range range_m, tci_range range_n,
                       tci_range_2d_func func, void* payload)
{
    (void)n;
    (void)idx;
    func(comm, 0, range_m.size, 0, range_n.size, payload);
}

#endif

void tci_comm_distribute_over_gangs(tci_comm* comm, tci_range range,
                                    tci_range_func func, void* payload)
{
    tci_distribute(comm->ngang, comm->gid, comm, range, func, payload);
}

void tci_comm_distribute_over_threads(tci_comm* comm, tci_range range,
                                      tci_range_func func, void* payload)
{
    tci_distribute(comm->nthread, comm->tid, tci_single, range, func, payload);
}

void tci_comm_distribute_over_gangs_2d(tci_comm* comm, tci_range range_m,
                                       tci_range range_n,
                                       tci_range_2d_func func, void* payload)
{
    tci_distribute_2d(comm->ngang, comm->gid, comm, range_m, range_n,
                      func, payload);
}

void tci_comm_distribute_over_threads_2d(tci_comm* comm, tci_range range_m,
                                         tci_range range_n,
                                         tci_range_2d_func func, void* payload)
{
    tci_distribute_2d(comm->nthread, comm->tid, tci_single, range_m, range_n,
                      func, payload);
}
