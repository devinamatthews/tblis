#ifndef _TENSOR_CORE_TENSOR_SLICER_HPP_
#define _TENSOR_CORE_TENSOR_SLICER_HPP_

#include "core/tensor_class.hpp"

namespace tensor
{

template <typename T>
class Slicer
{
    public:
        Slicer(Tensor<T>& A, const std::vector<gint_t>& dims)
        : first(true), dims(dims), A0(dims.size()), a1(std::max((size_t)1,dims.size())), A2(dims.size())
        {
            if (this->dims.size() == 0)
            {
                View(A, a1[0]);
                return;
            }

            std::sort(this->dims.begin(), this->dims.end(), std::greater<gint_t>());

            for (gint_t i = 0;i < this->dims.size();i++)
            {
                if (this->dims[i] < 0 || this->dims[i] >= A.getDimension()) abort();
                if (i > 0 && this->dims[i] == this->dims[i-1]) abort();
            }

            Slice(A, A0[0], a1[0], A2[0], this->dims[0], 0);
            for (gint_t i = 1;i < this->dims.size();i++)
            {
                Slice(a1[i-1], A0[i], a1[i], A2[i], this->dims[i], 0);
            }
        }

        Slicer(const Tensor<T>& A, const std::vector<gint_t>& dims)
        : first(true), dims(dims), A0(dims.size()), a1(std::max((size_t)1,dims.size())), A2(dims.size())
        {
            if (this->dims.size() == 0)
            {
                LockedView(A, a1[0]);
                return;
            }

            std::sort(this->dims.begin(), this->dims.end(), std::greater<gint_t>());

            for (gint_t i = 0;i < this->dims.size();i++)
            {
                if (this->dims[i] < 0 || this->dims[i] >= A.getDimension()) abort();
                if (i > 0 && this->dims[i] == this->dims[i-1]) abort();
            }

            LockedSlice(A, A0[0], a1[0], A2[0], this->dims[0], 0);
            for (gint_t i = 1;i < this->dims.size();i++)
            {
                LockedSlice(a1[i-1], A0[i], a1[i], A2[i], this->dims[i], 0);
            }
        }

        bool nextSlice(Tensor<T>& a)
        {
            if (first)
            {
                View(a1[std::max((size_t)1,dims.size())-1], a);
                first = false;
                return true;
            }
            else if (dims.size() != 0)
            {
                if (!nextSlice(dims.size()-1)) return false;
                View(a1[dims.size()-1], a);
                return true;
            }

            return false;
        }

    private:
        bool nextSlice(gint_t i)
        {
            if (A2[i].getLength(dims[i]) == 0)
            {
                if (i == 0 || !nextSlice(i-1))
                {
                    return false;
                }
                else
                {
                    Slice(a1[i-1], A0[i], a1[i], A2[i], dims[i], 0);
                    return true;
                }
            }

            UnsliceBack(A0[i], a1[i], A0[i], dims[i]);
            SliceFront(A2[i], a1[i], A2[i], dims[i]);

            return true;
        }

        bool first;
        std::vector<gint_t> dims;
        std::vector<Tensor<T> > A0;
        std::vector<Tensor<T> > a1;
        std::vector<Tensor<T> > A2;
};

}

#endif
