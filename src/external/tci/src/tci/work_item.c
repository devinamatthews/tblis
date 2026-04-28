#include "tci/work_item.h"
#include "tci/yield.h"

#include <stdlib.h>

int tci_work_item_try_work(tci_work_item* item)
{
    int expected = TCI_NOT_WORKED_ON;

    if (tci_atomic_compare_exchange(item, &expected, TCI_IN_PROGRESS, 1, TCI_ATOMIC_ACQUIRE, TCI_ATOMIC_ACQUIRE))
    {
        return TCI_RESERVED;
    }
    else
    {
        return expected;
    }
}

void tci_work_item_finish(tci_work_item* item)
{
    tci_atomic_store(item, TCI_FINISHED, TCI_ATOMIC_RELEASE);
}

int tci_work_item_status(tci_work_item* item)
{
    return tci_atomic_load(item, TCI_ATOMIC_ACQUIRE);
}

void tci_work_item_wait(tci_work_item* item)
{
    while (tci_atomic_load(item, TCI_ATOMIC_ACQUIRE) != TCI_FINISHED)
        tci_yield();
}
