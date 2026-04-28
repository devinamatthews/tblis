#ifndef _TCI_WORK_ITEM_H_
#define _TCI_WORK_ITEM_H_

#include "tci/tci_config.h"

#ifdef __cplusplus
extern "C" {
#endif

enum
{
    TCI_NOT_WORKED_ON,
    TCI_IN_PROGRESS,
    TCI_RESERVED,
    TCI_FINISHED
};

#define TCI_WORK_ITEM_INIT {TCI_NOT_WORKED_ON}

typedef TCI_ATOMIC int tci_work_item;

int tci_work_item_try_work(tci_work_item* item);

void tci_work_item_finish(tci_work_item* item);

int tci_work_item_status(tci_work_item* item);

void tci_work_item_wait(tci_work_item* item);

#ifdef __cplusplus
}
#endif

#endif
