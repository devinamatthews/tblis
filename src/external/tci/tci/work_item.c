#include "work_item.h"

#include <stdlib.h>

int tci_work_item_try_work(tci_work_item* item)
{
    int expected = TCI_NOT_WORKED_ON;

    if (__atomic_compare_exchange_n(item, &expected, TCI_IN_PROGRESS,
                                    1, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE))
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
    __atomic_store_n(item, TCI_FINISHED, __ATOMIC_RELEASE);
}

int tci_work_item_status(tci_work_item* item)
{
    return __atomic_load_n(item, __ATOMIC_ACQUIRE);
}

void tci_work_item_wait(tci_work_item* item)
{
    while (__atomic_load_n(item, __ATOMIC_ACQUIRE) != TCI_FINISHED)
        tci_yield();
}
