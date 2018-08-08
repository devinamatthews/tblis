#ifndef _TCI_TASK_SET_H_
#define _TCI_TASK_SET_H_

#include "tci_global.h"

#include "parallel.h"
#include "slot.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*tci_task_func)(tci_comm*, unsigned, void*);

typedef struct tci_task_set
{
    tci_comm* comm;
    tci_comm subcomm;
    tci_slot* slots;
    unsigned ntask;
} tci_task_set;

void tci_task_set_init(tci_task_set* set, tci_comm* comm, unsigned ntask,
                       uint64_t work);

void tci_task_set_destroy(tci_task_set* set);

int tci_task_set_visit(tci_task_set* set, tci_task_func func, unsigned task,
                       void* payload);

int tci_task_set_visit_all(tci_task_set* set, tci_task_func func,
                           void* payload);

#ifdef __cplusplus
}
#endif

#endif
