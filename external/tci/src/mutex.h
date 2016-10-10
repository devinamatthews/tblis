#ifndef _TCI_MUTEX_H_
#define _TCI_MUTEX_H_

#include "tci_config.hpp"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    unsigned char data[TCI_MUTEX_SIZE] __attribute__((__aligned__(8)));
} tci_mutex_t;

int tci_mutex_init(tci_mutex_t* mutex);

int tci_mutex_destroy(tci_mutex_t* mutex);

int tci_mutex_lock(tci_mutex_t* mutex);

int tci_mutex_trylock(tci_mutex_t* mutex);

int tci_mutex_unlock(tci_mutex_t* mutex);

extern void tci_printf(const char* fmt, ...);

#ifdef __cplusplus
}
#endif

#endif
