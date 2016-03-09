#ifndef _TBLIS_ITERATOR_HPP_
#define _TBLIS_ITERATOR_HPP_

#include "tblis.hpp"

#include "util/util.hpp"

#include <vector>
#include <iostream>

namespace tblis
{

class Iterator
{
    public:
        Iterator(const std::vector<dim_t>& len,
                 const std::vector<inc_t>& stride_0)
        : first(true), pos(len.size()), len(len), stride(1)
        {
            stride[0] = stride_0;
            check();
        }

        Iterator(const std::vector<dim_t>& len,
                 const std::vector<inc_t>& stride_0,
                 const std::vector<inc_t>& stride_1)
        : first(true), pos(len.size()), len(len), stride(2)
        {
            stride[0] = stride_0;
            stride[1] = stride_1;
            check();
        }

        Iterator(const std::vector<dim_t>& len,
                 const std::vector<inc_t>& stride_0,
                 const std::vector<inc_t>& stride_1,
                 const std::vector<inc_t>& stride_2)
        : first(true), pos(len.size()), len(len), stride(3)
        {
            stride[0] = stride_0;
            stride[1] = stride_1;
            stride[2] = stride_2;
            check();
        }

        template <typename cv_ptr_0>
        bool nextIteration(cv_ptr_0& ptr_0)
        {
            if (stride.size() != 1) abort();

            if (first)
            {
                first = false;
            }
            else
            {
                if (len.size() == 0)
                {
                    first = true;
                    return false;
                }

                for (gint_t i = 0;i < len.size();i++)
                {
                    if (pos[i] == len[i]-1)
                    {
                        ptr_0 -= pos[i]*stride[0][i];
                        pos[i] = 0;

                        if (i == len.size()-1)
                        {
                            first = true;
                            return false;
                        }
                    }
                    else
                    {
                        ptr_0 += stride[0][i];
                        pos[i]++;

                        return true;
                    }
                }
            }

            return true;
        }

        template <typename cv_ptr_0, typename cv_ptr_1>
        bool nextIteration(cv_ptr_0& ptr_0, cv_ptr_1& ptr_1)
        {
            if (stride.size() != 2) abort();

            if (first)
            {
                first = false;
            }
            else
            {
                if (len.size() == 0)
                {
                    first = true;
                    return false;
                }

                for (gint_t i = 0;i < len.size();i++)
                {
                    if (pos[i] == len[i]-1)
                    {
                        ptr_0 -= pos[i]*stride[0][i];
                        ptr_1 -= pos[i]*stride[1][i];
                        pos[i] = 0;

                        if (i == len.size()-1)
                        {
                            first = true;
                            return false;
                        }
                    }
                    else
                    {
                        ptr_0 += stride[0][i];
                        ptr_1 += stride[1][i];
                        pos[i]++;

                        return true;
                    }
                }
            }

            return true;
        }

        template <typename cv_ptr_0, typename cv_ptr_1, typename cv_ptr_2>
        bool nextIteration(cv_ptr_0& ptr_0, cv_ptr_1& ptr_1, cv_ptr_2& ptr_2)
        {
            if (stride.size() != 3) abort();

            if (first)
            {
                first = false;
            }
            else
            {
                if (len.size() == 0)
                {
                    first = true;
                    return false;
                }

                for (gint_t i = 0;i < len.size();i++)
                {
                    if (pos[i] == len[i]-1)
                    {
                        ptr_0 -= pos[i]*stride[0][i];
                        ptr_1 -= pos[i]*stride[1][i];
                        ptr_2 -= pos[i]*stride[2][i];
                        pos[i] = 0;

                        if (i == len.size()-1)
                        {
                            first = true;
                            return false;
                        }
                    }
                    else
                    {
                        ptr_0 += stride[0][i];
                        ptr_1 += stride[1][i];
                        ptr_2 += stride[2][i];
                        pos[i]++;

                        return true;
                    }
                }
            }

            return true;
        }

    private:
        void check()
        {
            for (auto& s : stride)
            {
                ASSERT(s.size() == len.size());
            }

            for (dim_t l : len)
            {
                ASSERT(l > 0);
            }
        }

        bool first;
        std::vector<dim_t> pos;
        std::vector<dim_t> len;
        std::vector<std::vector<inc_t> > stride;
};

}

#endif
