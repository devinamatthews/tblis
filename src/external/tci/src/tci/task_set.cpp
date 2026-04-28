#include "tci/task_set.h"

#if TCI_USE_TBB_THREADS

#include <tbb/tbb.h>

extern "C"
void tci_task_set_init(tci_task_set* set, tci_comm* comm, unsigned ntask, uint64_t work)
{
    set->comm = (tci_comm*)new tbb::task_group();
    set->ntask = ntask;
    set->slots = new tci_slot[ntask];
    for (unsigned task = 0;task < ntask;task++)
        tci_slot_init(set->slots+task, 0);

    unsigned nt = comm->nthread;
    unsigned nt_outer, nt_inner;
    tci_partition_2x2(nt, work, (work == 0 ? 1 : nt),
                      ntask, ntask, &nt_inner, &nt_outer);
    tci_comm_gang(comm, &set->subcomm, TCI_EVENLY, nt_outer, 0);
}

extern "C"
void tci_task_set_destroy(tci_task_set* set)
{
    ((tbb::task_group*)set->comm)->wait();
    delete[] set->slots;
    delete (tbb::task_group*)set->comm;
}

extern "C"
int tci_task_set_visit(tci_task_set* set, tci_task_func func, unsigned task,
                       void* payload)
{
    if (task > set->ntask) return EINVAL;
    if (!tci_slot_try_fill(set->slots+task, 0, 1)) return EALREADY;

    ((tbb::task_group*)set->comm)->run(
    [set,func,task,payload]
    {
        func(&set->subcomm, task, payload);
    });

    return 0;
}

#elif TCI_USE_PPL_THREADS

#include <ppl.h>

extern "C"
void tci_task_set_init(tci_task_set* set, tci_comm* comm, unsigned ntask, uint64_t work)
{
    (void)comm;
    (void)work;

    set->comm = (tci_comm*)new concurrency::task_group();
    set->ntask = ntask;
    set->slots = new tci_slot[ntask];
    for (unsigned task = 0;task < ntask;task++)
        tci_slot_init(set->slots+task, 0);
}

extern "C"
void tci_task_set_destroy(tci_task_set* set)
{
    ((concurrency::task_group*)set->comm)->wait();
    delete[] set->slots;
    delete (concurrency::task_group*)set->comm;
}

extern "C"
int tci_task_set_visit(tci_task_set* set, tci_task_func func, unsigned task,
                       void* payload)
{
    if (task > set->ntask) return EINVAL;
    if (!tci_slot_try_fill(set->slots+task, 0, 1)) return EALREADY;

    ((concurrency::task_group*)set->comm)->run(
    [&,func,task,payload]
    {
        func(tci_single, task, payload);
    });

    return 0;
}

#endif
