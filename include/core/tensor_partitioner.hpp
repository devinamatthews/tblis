#ifndef _TENSOR_CORE_TENSOR_PARTITIONER_HPP_
#define _TENSOR_CORE_TENSOR_PARTITIONER_HPP_

#include "core/tensor_class.hpp"

namespace tensor
{

template <typename T>
class Partitioner
{
    public:
        Partitioner(Tensor<T>& A, const std::vector<gint_t>& dims,
                                  const std::vector<dim_t>& bs)
        : first(true), dims(dims), bs(bs), A0(dims.size()), A1(std::max((size_t)1,dims.size())), A2(dims.size())
        {
            if (this->dims.size() == 0)
            {
                View(A, A1[0]);
                return;
            }

            std::sort(this->dims.begin(), this->dims.end(), std::greater<gint_t>());

            if (this->dims.size() != this->bs.size()) abort();

            for (gint_t i = 0;i < this->dims.size();i++)
            {
                if (this->bs[i] <= 0) abort();
                if (this->dims[i] < 0 || this->dims[i] >= A.getDimension()) abort();
                if (i > 0 && this->dims[i] == this->dims[i-1]) abort();
            }

            Partition(    A, A0[0], A1[0], this->dims[0],           0);
            Partition(A1[0], A1[0], A2[0], this->dims[0], this->bs[0]);
            for (gint_t i = 1;i < this->dims.size();i++)
            {
                Partition(A1[i-1], A0[i], A1[i], this->dims[i],           0);
                Partition(A1[  i], A1[i], A2[i], this->dims[i], this->bs[i]);
            }
        }

        Partitioner(const Tensor<T>& A, const std::vector<gint_t>& dims,
                                        const std::vector<dim_t>& bs)
        : first(true), dims(dims), bs(bs), A0(dims.size()), A1(std::max((size_t)1,dims.size())), A2(dims.size())
        {
            if (this->dims.size() == 0)
            {
                LockedView(A, A1[0]);
                return;
            }

            std::sort(this->dims.begin(), this->dims.end(), std::greater<gint_t>());

            if (this->dims.size() != this->bs.size()) abort();

            for (gint_t i = 0;i < this->dims.size();i++)
            {
                if (this->bs[i] <= 0) abort();
                if (this->dims[i] < 0 || this->dims[i] >= A.getDimension()) abort();
                if (i > 0 && this->dims[i] == this->dims[i-1]) abort();
            }

            LockedPartition(    A, A0[0], A1[0], this->dims[0],           0);
            LockedPartition(A1[0], A1[0], A2[0], this->dims[0], this->bs[0]);
            for (gint_t i = 1;i < this->dims.size();i++)
            {
                LockedPartition(A1[i-1], A0[i], A1[i], this->dims[i],           0);
                LockedPartition(A1[  i], A1[i], A2[i], this->dims[i], this->bs[i]);
            }
        }

        bool nextPartition(Tensor<T>& a)
        {
            if (first)
            {
                View(A1[std::max((size_t)1,dims.size()-1)], a);
                first = false;
                return true;
            }
            else if (dims.size() != 0)
            {
                if (!nextPartition(dims.size()-1)) return false;
                View(A1[dims.size()-1], a);
                return true;
            }

            return false;
        }

    private:
        bool nextPartition(gint_t i)
        {
            if (A2[i].getLength(dims[i]) == 0)
            {
                if (i == 0 || !nextPartition(i-1))
                {
                    return false;
                }
                else
                {
                    Partition(A1[i-1], A0[i], A1[i], dims[i],     0);
                    Partition(A1[  i], A1[i], A2[i], dims[i], bs[i]);
                    return true;
                }
            }

            Unpartition(A0[i], A1[i], A0[i], dims[i]);
            Partition(A2[i], A1[i], A2[i], dims[i], bs[i]);

            return true;
        }

        bool first;
        std::vector<gint_t> dims;
        std::vector<dim_t> bs;
        std::vector<Tensor<T> > A0;
        std::vector<Tensor<T> > A1;
        std::vector<Tensor<T> > A2;
};

}

#endif
