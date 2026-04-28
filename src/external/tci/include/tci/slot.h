#ifndef _TCI_SLOT_H_
#define _TCI_SLOT_H_

#include "tci/tci_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef TCI_ATOMIC int tci_slot;

int tci_slot_init(tci_slot* slot, int empty);

int tci_slot_is_filled(tci_slot* slot, int empty);

int tci_slot_try_fill(tci_slot* slot, int empty, int value);

void tci_slot_fill(tci_slot* slot, int empty, int value);

void tci_slot_clear(tci_slot* slot, int empty);

#ifdef __cplusplus
}
#endif

#endif
