#ifndef _TCI_WORK_ITEM_H_
#define _TCI_WORK_ITEM_H_

#include "tci_global.h"

#include "yield.h"

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

typedef volatile int tci_work_item;

int tci_work_item_try_work(tci_work_item* item);

void tci_work_item_finish(tci_work_item* item);

int tci_work_item_status(tci_work_item* item);

void tci_work_item_wait(tci_work_item* item);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TCI_DONT_USE_CXX11)

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

#endif
