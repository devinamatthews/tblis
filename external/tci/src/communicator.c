#include "tci.h"

#include <stdlib.h>
#include <string.h>

int tci_comm_init_single(tci_comm_t* comm)
{
    comm->context = NULL;
    comm->nthread = 1;
    comm->tid = 0;
    comm->ngang = 1;
    comm->gid = 0;
    return 0;
}

int tci_comm_init(tci_comm_t* comm, tci_context_t* context,
                  int tid, int ngang, int gid)
{
    comm->context = context;
    comm->tid = tid;
    comm->ngang = ngang;
    comm->gid = gid;

    if (context)
    {
        tci_context_attach(comm->context);
        comm->nthread = context->barrier.nthread;
    }
    else
    {
        comm->nthread = 1;
    }

    return 0;
}

int tci_comm_destroy(tci_comm_t* comm)
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

int tci_comm_is_master(const tci_comm_t* comm)
{
    return comm->tid == 0;
}

int tci_comm_barrier(tci_comm_t* comm)
{
    if (comm->nthread == 1) return 0;
    return tci_context_barrier(comm->context, comm->tid);
}

int tci_comm_bcast(tci_comm_t* comm, void** object, int root)
{
    if (comm->nthread == 1) return 0;

    if (comm->tid == root)
    {
        return tci_context_send(comm->context, comm->tid, *object);
    }
    else
    {
        return tci_context_receive(comm->context, comm->tid, object);
    }
}

int tci_comm_bcast_nowait(tci_comm_t* comm, void** object, int root)
{
    if (comm->nthread == 1) return 0;

    if (comm->tid == root)
    {
        return tci_context_send_nowait(comm->context, comm->tid, *object);
    }
    else
    {
        return tci_context_receive_nowait(comm->context, comm->tid, object);
    }
}

int tci_comm_gang(tci_comm_t* parent, tci_comm_t* child, int n, int block,
                  int new_tid, int new_nthread)
{
    tci_context_t* contexts_buf[n];
    tci_context_t** contexts = &contexts_buf[0];

    memset(contexts_buf, 0, sizeof(contexts_buf));
    tci_comm_bcast_nowait(parent, (void**)&contexts, 0);

    if (new_tid == 0 && new_nthread > 1)
    {
        tci_context_init(&contexts[block], new_nthread,
                         parent->context->barrier.group_size);
    }

    tci_comm_barrier(parent);

    tci_comm_init(child, contexts[block], new_tid, n, block);

    tci_comm_barrier(parent);

    return 0;
}

int tci_comm_gang_evenly(tci_comm_t* parent, tci_comm_t* child, int n)
{
    int nt = parent->nthread;
    int tid = parent->tid;

    if (n >= nt) return tci_comm_init(child, NULL, 0, nt, tid);

    int block = (n*tid)/nt;
    int block_first = (block*nt)/n;
    int block_last = ((block+1)*nt)/n;
    int new_tid = tid-block_first;
    int new_nthread = block_last-block_first;

    return tci_comm_gang(parent, child, n, block, new_tid, new_nthread);
}

int tci_comm_gang_block_cyclic(tci_comm_t* parent, tci_comm_t* child, int n, int bs)
{
    int nt = parent->nthread;
    int tid = parent->tid;

    if (n >= nt) return tci_comm_init(child, NULL, 0, nt, tid);

    int block = (tid/bs)%n;
    int nsubblock_tot = nt/bs;
    int nsubblock = nsubblock_tot/n;
    int new_tid = ((tid/bs)/n)*bs + (tid%bs);
    int new_nthread = nsubblock*bs + TCI_MIN(bs, nt-nsubblock*n*bs-block*bs);

    return tci_comm_gang(parent, child, n, block, new_tid, new_nthread);
}

int tci_comm_gang_blocked(tci_comm_t* parent, tci_comm_t* child, int n)
{
    int nt = parent->nthread;
    int tid = parent->tid;

    if (n >= nt) return tci_comm_init(child, NULL, 0, nt, tid);

    int bs = (nt+n-1)/n;
    int block = tid/bs;
    int new_tid = tid-block*bs;
    int new_nthread = TCI_MIN(bs, nt-block*bs);

    return tci_comm_gang(parent, child, n, block, new_tid, new_nthread);
}

int tci_comm_gang_cyclic(tci_comm_t* parent, tci_comm_t* child, int n)
{
    int nt = parent->nthread;
    int tid = parent->tid;

    if (n >= nt) return tci_comm_init(child, NULL, 0, nt, tid);

    int block = tid%n;
    int new_tid = tid/n;
    int new_nthread = (nt-block+n-1)/n;

    return tci_comm_gang(parent, child, n, block, new_tid, new_nthread);
}

void tci_distribute(int n, int idx, int64_t range, int64_t granularity,
                    int64_t* first, int64_t* last, int64_t* max)
{
    int64_t ngrain = (range+granularity-1)/granularity;
    int64_t max_size = ((ngrain+n-1)/n)*granularity;

    if (first) *first =         (( idx   *ngrain)/n)*granularity;
    if ( last)  *last = TCI_MIN((((idx+1)*ngrain)/n)*granularity, range);
    if (  max)   *max =         (( idx   *ngrain)/n)*granularity+max_size;
}

void tci_comm_distribute_over_gangs(tci_comm_t* comm, int64_t range,
                                    int64_t granularity, int64_t* first,
                                    int64_t* last, int64_t* max)
{
    tci_distribute(comm->ngang, comm->gid, range, granularity,
                   first, last, max);
}

void tci_comm_distribute_over_threads(tci_comm_t* comm, int64_t range,
                                      int64_t granularity, int64_t* first,
                                      int64_t* last, int64_t* max)
{
    tci_distribute(comm->nthread, comm->tid, range, granularity,
                   first, last, max);
}

void tci_comm_distribute_over_gangs_2d(tci_comm_t* comm, int64_t range_m,
                                       int64_t range_n, int64_t granularity_m,
                                       int64_t granularity_n, int64_t* first_m,
                                       int64_t* last_m, int64_t* max_m,
                                       int64_t* first_n, int64_t* last_n,
                                       int64_t* max_n)
{
    int m, n;
    tci_partition_2x2(comm->ngang, range_m, range_n, &m, &n);

    int idx_m = comm->gid % m;
    int idx_n = comm->gid / m;

    tci_distribute(m, idx_m, range_m, granularity_m, first_m, last_m, max_m);
    tci_distribute(n, idx_n, range_n, granularity_n, first_n, last_n, max_n);
}

void tci_comm_distribute_over_threads_2d(tci_comm_t* comm, int64_t range_m,
                                         int64_t range_n, int64_t granularity_m,
                                         int64_t granularity_n, int64_t* first_m,
                                         int64_t* last_m, int64_t* max_m,
                                         int64_t* first_n, int64_t* last_n,
                                         int64_t* max_n)
{
    int m, n;
    tci_partition_2x2(comm->nthread, range_m, range_n, &m, &n);

    int idx_m = comm->tid % m;
    int idx_n = comm->tid / m;

    tci_distribute(m, idx_m, range_m, granularity_m, first_m, last_m, max_m);
    tci_distribute(n, idx_n, range_n, granularity_n, first_n, last_n, max_n);
}
