#ifndef _TCI_WORK_ITEM_HPP_
#define _TCI_WORK_ITEM_HPP_

#include "tci/work_item.h"

namespace tci
{

class work_item
{
    public:
        work_item() {}

        work_item(const work_item&) = delete;

        work_item& operator=(const work_item&) = delete;

        int status() const
        {
            return tci_work_item_status(const_cast<work_item&>(*this));
        }

        int try_work()
        {
            return tci_work_item_try_work(*this);
        }

        void finish()
        {
            tci_work_item_finish(*this);
        }

        void wait() const
        {
            tci_work_item_wait(const_cast<work_item&>(*this));
        }

        operator tci_work_item*() { return &_item; }

        operator const tci_work_item*() const { return &_item; }

    protected:
        tci_work_item _item = TCI_WORK_ITEM_INIT;
};

}

#endif
