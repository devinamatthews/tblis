#include "slot.h"

int tci_slot_init(tci_slot* slot, int empty)
{
    *slot = empty;
    return 0;
}

int tci_slot_is_filled(tci_slot* slot, int empty)
{
    return __atomic_load_n(slot, __ATOMIC_ACQUIRE) != empty;
}

int tci_slot_try_fill(tci_slot* slot, int empty, int value)
{
    if (__atomic_compare_exchange_n(slot, &empty, value, 1,
                                    __ATOMIC_ACQUIRE,
                                    __ATOMIC_RELAXED)) return 1;

    return empty == value;
}

void tci_slot_fill(tci_slot* slot, int empty, int value)
{
    while (true)
    {
        int expected = empty;
        if (__atomic_compare_exchange_n(slot, &expected, value, 0,
                                        __ATOMIC_ACQUIRE,
                                        __ATOMIC_RELAXED)) break;
        tci_yield();
    }
}

void tci_slot_clear(tci_slot* slot, int empty)
{
    __atomic_store_n(slot, empty, __ATOMIC_RELEASE);
}
