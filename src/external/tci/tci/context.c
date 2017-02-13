#include "tci.h"

#include <stdlib.h>
#include <errno.h>

int tci_context_init(tci_context** context,
                     unsigned nthread, unsigned group_size)
{
    *context = (tci_context*)malloc(sizeof(tci_context));
    if (!*context) return ENOMEM;
    (*context)->refcount = 0;
    (*context)->buffer = NULL;
    return tci_barrier_init(&(*context)->barrier, nthread, group_size);
}

int tci_context_attach(tci_context* context)
{
    __atomic_fetch_add(&context->refcount, 1, __ATOMIC_RELAXED);
    return 0;
}

int tci_context_detach(tci_context* context)
{
    if (__atomic_sub_fetch(&context->refcount, 1, __ATOMIC_RELEASE) == 0)
    {
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        int ret = tci_barrier_destroy(&context->barrier);
        free(context);
        return ret;
    }
    return 0;
}

int tci_context_barrier(tci_context* context, unsigned tid)
{
    return tci_barrier_wait(&context->barrier, tid);
}

int tci_context_send(tci_context* context, unsigned tid, void* object)
{
    context->buffer = object;
    int ret = tci_context_barrier(context, tid);
    if (ret != 0) return ret;
    return tci_context_barrier(context, tid);
}

int tci_context_send_nowait(tci_context* context,
                            unsigned tid, void* object)
{
    context->buffer = object;
    return tci_context_barrier(context, tid);
}

int tci_context_receive(tci_context* context, unsigned tid, void** object)
{
    int ret = tci_context_barrier(context, tid);
    if (ret != 0) return ret;
    *object = context->buffer;
    return tci_context_barrier(context, tid);
}

int tci_context_receive_nowait(tci_context* context,
                               unsigned tid, void** object)
{
    int ret = tci_context_barrier(context, tid);
    if (ret != 0) return ret;
    *object = context->buffer;
    return 0;
}
