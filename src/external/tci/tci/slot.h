#ifndef _TCI_SLOT_H_
#define _TCI_SLOT_H_

#include "tci_config.h"

#include "yield.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef volatile int tci_slot;

int tci_slot_init(tci_slot* slot, int empty)
{
    slot = empty;
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

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TCI_DONT_USE_CXX11)

namespace tci
{

template <int Empty=-1>
class slot
{
    public:
        slot() {}

        slot(const slot&) = delete;

        slot& operator=(const slot&) = delete;

        bool is_filled() const
        {
            return tci_slot_is_filled(*this, Empty);
        }

        bool try_fill(int value)
        {
            return tci_slot_try_fill(*this, Empty, value);
        }

        void fill(int value)
        {
            tci_slot_fill(*this, Empty, value);
        }

        void clear()
        {
            tci_slot_clear(*this, Empty);
        }

        operator tci_slot*() { return &_slot; }

        operator const tci_slot*() const { return &_slot; }

    protected:
        tci_slot _slot = {Empty};
};

}

#endif

#endif
