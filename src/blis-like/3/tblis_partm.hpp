#ifndef _TENSOR_TBLIS_PARTITION_HPP_
#define _TENSOR_TBLIS_PARTITION_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <template <typename> class MT, int Dim>
struct Partition
{
    template <typename T, typename Child, typename... Children>
    struct run
    {
        template <typename MatrixA, typename MatrixB, typename MatrixC>
        void operator()(T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C) const
        {
            using namespace matrix_constants;

            constexpr dim_t M_def = MT<T>::def;
            constexpr dim_t M_max = MT<T>::max;

            dim_t m_u = (Dim == DIM_M ? A.length() : Dim == DIM_N ? B.width() : A.width());
            dim_t m_v = (Dim == DIM_M ? C.length() : Dim == DIM_N ? C.width() : B.length());

            dim_t u = 0;
            dim_t v = 0;
            for (dim_t off_u = 0, off_v = 0;
                 off_u < m_u && off_v < m_v;
                 off_u += u, off_v += v)
            {
                if (m_u-off_u <= M_max && m_v-off_v <= M_max)
                {
                    u = m_u-off_u;
                    v = m_v-off_v;
                }
                else
                {
                    u = M_def;
                    v = M_def;
                }

                (Dim == DIM_M ? A.length(u) : Dim == DIM_N ? B.width(u) : A.width(u));
                (Dim == DIM_M ? C.length(v) : Dim == DIM_N ? C.width(v) : B.length(v));

                typename Child::template run<T, Children...>()(alpha, A, B, beta, C);

                (Dim == DIM_M ? A.shift_down() : Dim == DIM_N ? B.shift_right() : A.shift_right());
                (Dim == DIM_M ? C.shift_down() : Dim == DIM_N ? C.shift_right() : B.shift_down());

                if (Dim == DIM_K) beta = 1.0;
            }

            (Dim == DIM_M ? A.length(m_u) : Dim == DIM_N ? B.width(m_u)   : A.width(m_u));
            (Dim == DIM_M ? C.length(m_v) : Dim == DIM_N ? C.width(m_v)   : B.length(m_v));
            (Dim == DIM_M ? A.shift_up()  : Dim == DIM_N ? B.shift_left() : A.shift_left());
            (Dim == DIM_M ? C.shift_up()  : Dim == DIM_N ? C.shift_left() : B.shift_up());
        }
    };
};

template <template <typename> class MT>
using PartitionM = Partition<MT,matrix_constants::DIM_M>;

template <template <typename> class NT>
using PartitionN = Partition<NT,matrix_constants::DIM_N>;

template <template <typename> class KT>
using PartitionK = Partition<KT,matrix_constants::DIM_K>;

}
}

#endif
