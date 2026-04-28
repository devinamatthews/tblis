#include "tci/slot.h"
#include "tci/yield.h"

int tci_slot_init(tci_slot* slot, int empty)
{
    *slot = empty;
    return 0;
}

int tci_slot_is_filled(tci_slot* slot, int empty)
{
    return tci_atomic_load(slot, TCI_ATOMIC_ACQUIRE) != empty;
}

int tci_slot_try_fill(tci_slot* slot, int empty, int value)
{
    if (tci_atomic_compare_exchange(slot, &empty, value, 1,
                                    TCI_ATOMIC_ACQUIRE,
                                    TCI_ATOMIC_RELAXED)) return 1;

    return empty == value;
}

void tci_slot_fill(tci_slot* slot, int empty, int value)
{
    while (true)
    {
        int expected = empty;
        if (tci_atomic_compare_exchange(slot, &expected, value, 0,
                                        TCI_ATOMIC_ACQUIRE,
                                        TCI_ATOMIC_RELAXED)) break;
        tci_yield();
    }
}

void tci_slot_clear(tci_slot* slot, int empty)
{
    tci_atomic_store(slot, empty, TCI_ATOMIC_RELEASE);
}
