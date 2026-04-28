#include "tci/pipeline.h"

void tci_pipeline_init(tci_pipeline** p, unsigned depth, size_t size, void* buffer)
{
    *p = (tci_pipeline*)malloc(sizeof(tci_pipeline) + sizeof(int)*(depth-1));

    (*p)->buffer = buffer;
    (*p)->size = size;
    (*p)->depth = depth;
    (*p)->last_filled = -1;
    (*p)->last_drained = -1;

    for (unsigned i = 0;i < depth;i++)
    {
        (*p)->state[i] = TCI_NOT_FILLED;
    }
}

void tci_pipeline_destroy(tci_pipeline* p)
{
    free(p);
}

void* tci_pipeline_drain(tci_pipeline* p)
{
    //TODO
}

int tci_pipeline_trydrain(tci_pipeline* p, void** buffer)
{
    //TODO
}

void tci_pipeline_drained(tci_pipeline* p, void* buffer)
{
    //TODO
}

void* tci_pipeline_fill(tci_pipeline* p)
{
    //TODO
}

int tci_pipeline_tryfill(tci_pipeline* p, void** buffer)
{
    //TODO
}

void tci_pipeline_filled(tci_pipeline* p, void* buffer)
{
    //TODO
}
