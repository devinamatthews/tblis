#ifndef _TCI_BARRIER_H_
#define _TCI_BARRIER_H_

#include "tci/tci_config.h"
#include "tci/mutex.h"

#if TCI_USE_PTHREAD_BARRIER

#include <pthread.h>

typedef struct tci_barrier_node
{
    struct tci_barrier_node* parent;
    pthread_barrier_t barrier;
} tci_barrier_node;

#else

typedef struct tci_barrier_node
{
    struct tci_barrier_node* parent;
    unsigned nchildren;
    TCI_ATOMIC unsigned step;
    TCI_ATOMIC unsigned nwaiting;
} tci_barrier_node;

typedef struct tci_barrier
{
    union
    {
        tci_barrier_node* array;
        tci_barrier_node single;
    } barrier;
    unsigned nthread;
    unsigned group_size;
    int is_tree;
} tci_barrier;
#endif

#ifdef __cplusplus
extern "C" {
#endif

int tci_barrier_node_init(tci_barrier_node* barrier,
                          tci_barrier_node* parent,
                          unsigned nchildren);

int tci_barrier_node_destroy(tci_barrier_node* barrier);

int tci_barrier_node_wait(tci_barrier_node* barrier);

int tci_barrier_is_tree(tci_barrier* barrier);

int tci_barrier_init(tci_barrier* barrier,
                     unsigned nthread,
                     unsigned group_size);

int tci_barrier_destroy(tci_barrier* barrier);

int tci_barrier_wait(tci_barrier* barrier,
                     unsigned tid);

#ifdef __cplusplus
}
#endif

#endif
