#ifndef _TCI_SLOT_HPP_
#define _TCI_SLOT_HPP_

#include "tci/slot.h"

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
