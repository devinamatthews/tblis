#ifndef _TCI_CONTEXT_H_
#define _TCI_CONTEXT_H_

#include "tci_global.h"

#include "barrier.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tci_context
{
    tci_barrier barrier;
    volatile void* buffer;
    volatile unsigned refcount;
} tci_context;

int tci_context_init(tci_context** context,
                     unsigned nthread, unsigned group_size);

int tci_context_attach(tci_context* context);

int tci_context_detach(tci_context* context);

int tci_context_barrier(tci_context* context, unsigned tid);

int tci_context_send(tci_context* context, unsigned tid, void* object);

int tci_context_send_nowait(tci_context* context,
                            unsigned tid,void* object);

int tci_context_receive(tci_context* context, unsigned tid, void** object);

int tci_context_receive_nowait(tci_context* context,
                               unsigned tid, void** object);

#ifdef __cplusplus
}
#endif

#endif
