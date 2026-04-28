#ifndef _TCI_PIPELINE_H_
#define _TCI_PIPELINE_H_

#include "tci/tci_config.h"

#ifdef __cplusplus
extern "C" {
#endif

enum
{
    TCI_NOT_FILLED,
    TCI_DRAINING,
    TCI_FILLING,
    TCI_FILLED
};

typedef struct tci_pipeline
{
    void* buffer;
    size_t size;
    unsigned depth;
    unsigned last_drained;
    unsigned last_filled;
    TCI_ATOMIC int state[1];
} tci_pipeline;

void tci_pipeline_init(tci_pipeline** p, unsigned depth, size_t size, void* buffer);

void tci_pipeline_destroy(tci_pipeline* p);

void* tci_pipeline_drain(tci_pipeline* p);

int tci_pipeline_trydrain(tci_pipeline* p, void** buffer);

void tci_pipeline_drained(tci_pipeline* p, void* buffer);

void* tci_pipeline_fill(tci_pipeline* p);

int tci_pipeline_tryfill(tci_pipeline* p, void** buffer);

void tci_pipeline_filled(tci_pipeline* p, void* buffer);

#ifdef __cplusplus
}
#endif

#endif
