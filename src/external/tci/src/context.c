#include "tci.h"

#include <stdlib.h>

int tci_context_init(tci_context_t** context,
                     unsigned nthread, unsigned group_size)
{
    *context = (tci_context_t*)malloc(sizeof(tci_context_t));
    if (!*context) return ENOMEM;
    (*context)->refcount = 0;
    (*context)->buffer = NULL;
    return tci_barrier_init(&(*context)->barrier, nthread, group_size);
}

int tci_context_attach(tci_context_t* context)
{
    __sync_fetch_and_add(&context->refcount, 1);
    return 0;
}

int tci_context_detach(tci_context_t* context)
{
    if (__sync_sub_and_fetch(&context->refcount, 1) == 0)
    {
        int ret = tci_barrier_destroy(&context->barrier);
        free(context);
        return ret;
    }
    return 0;
}

int tci_context_barrier(tci_context_t* context, unsigned tid)
{
    return tci_barrier_wait(&context->barrier, tid);
}

int tci_context_send(tci_context_t* context, unsigned tid, void* object)
{
    context->buffer = object;
    int ret = tci_context_barrier(context, tid);
    if (ret != 0) return ret;
    return tci_context_barrier(context, tid);
}

int tci_context_send_nowait(tci_context_t* context,
                            unsigned tid, void* object)
{
    context->buffer = object;
    return tci_context_barrier(context, tid);
}

int tci_context_receive(tci_context_t* context, unsigned tid, void** object)
{
    int ret = tci_context_barrier(context, tid);
    if (ret != 0) return ret;
    *object = context->buffer;
    return tci_context_barrier(context, tid);
}

int tci_context_receive_nowait(tci_context_t* context,
                               unsigned tid, void** object)
{
    int ret = tci_context_barrier(context, tid);
    if (ret != 0) return ret;
    *object = context->buffer;
    return 0;
}
