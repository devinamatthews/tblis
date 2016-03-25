#ifndef _TBLIS_GEMM_UKR_HPP_
#define _TBLIS_GEMM_UKR_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T, dim_t MR, dim_t NR>
void AccumulateMicroTile(dim_t m, dim_t n, const T* p_ab,
                         T beta, T* p_c, inc_t rs_c, inc_t cs_c);

template <typename T, dim_t MR, dim_t NR>
void AccumulateMicroTile(dim_t m, dim_t n, const T* p_ab,
                         T beta, T* p_c, const inc_t* rs_c, inc_t cs_c);

template <typename T, dim_t MR, dim_t NR>
void AccumulateMicroTile(dim_t m, dim_t n, const T* p_ab,
                         T beta, T* p_c, inc_t rs_c, const inc_t* cs_c);

template <typename T, dim_t MR, dim_t NR>
void AccumulateMicroTile(dim_t m, dim_t n, const T* p_ab,
                         T beta, T* p_c, const inc_t* rs_c, const inc_t* cs_c);

template <typename T, dim_t MR, dim_t NR>
void GenericMicroKernel(dim_t k,
                        T alpha, const T* p_a, const T* p_b,
                        T beta, T* p_c, inc_t rs_c, inc_t cs_c,
                        const auxinfo_t* data);

template <template <typename> class MT, template <typename> class NT>
struct MicroKernel
{
    template <typename T>
    struct run
    {
        typedef basic_type_t<T> Tb;
        constexpr static dim_t MR = MT<T>::def;
        constexpr static dim_t NR = NT<T>::def;

        static Tb* fwd(const T* value)
        {
            return const_cast<Tb*>(reinterpret_cast<const Tb*>(value));
        }

        static Tb* fwd(const T& value)
        {
            return fwd(&value);
        }

        void operator()(ThreadCommunicator& comm, T alpha, Matrix<T>& A, Matrix<T>& B,
                         T beta, Matrix<T>& C) const;

        void operator()(ThreadCommunicator& comm, T alpha, Matrix<T>& A, Matrix<T>& B,
                         T beta, ScatterMatrix<T>& C) const;

        void operator()(ThreadCommunicator& comm, T alpha, Matrix<T>& A, Matrix<T>& B,
                         T beta, BlockScatterMatrix<T,MR,NR>& C) const;
    };
};

struct Noop
{
    template <typename T>
    struct run
    {
        void operator()(ThreadCommunicator& comm, T alpha, Matrix<T>& A, Matrix<T>& B, T beta, Matrix<T>& C) const
        {}

        void operator()(ThreadCommunicator& comm, T alpha, Matrix<T>& A, Matrix<T>& B, T beta, ScatterMatrix<T>& C) const
        {}
    };
};

}
}

#endif
