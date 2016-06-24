#ifndef _TBLIS_GEMM_UKR_HPP_
#define _TBLIS_GEMM_UKR_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T, dim_t MR, dim_t NR>
void AccumulateMicroTile(dim_t m, dim_t n, const T* restrict p_ab,
                         T beta, T* restrict p_c, inc_t rs_c, inc_t cs_c);

template <typename T, dim_t MR, dim_t NR>
void AccumulateMicroTile(dim_t m, dim_t n, const T* restrict p_ab,
                         T beta, T* restrict p_c,
                         const inc_t* restrict rs_c, inc_t cs_c);

template <typename T, dim_t MR, dim_t NR>
void AccumulateMicroTile(dim_t m, dim_t n, const T* restrict p_ab,
                         T beta, T* restrict p_c,
                         inc_t rs_c, const inc_t* restrict cs_c);

template <typename T, dim_t MR, dim_t NR>
void AccumulateMicroTile(dim_t m, dim_t n, const T* restrict p_ab,
                         T beta, T* restrict p_c,
                         const inc_t* restrict rs_c,
                         const inc_t* restrict cs_c);

template <typename T, dim_t MR, dim_t NR>
void GenericMicroKernel(dim_t k,
                        const T* restrict alpha,
                        const T* restrict a, const T* restrict b,
                        const T* restrict beta,
                        T* restrict c, inc_t rs_c, inc_t cs_c,
                        const void* restrict data, const void* restrict cntx);

template <typename Config>
struct MicroKernel
{
    template <typename T>
    struct run
    {
        constexpr static dim_t MR = Config::template MR<T>::def;
        constexpr static dim_t NR = Config::template NR<T>::def;
        constexpr static gemm_ukr_t<T> ukr = Config::template gemm_ukr<T>::value;

        void operator()(ThreadCommunicator& comm,
                        T alpha, const const_matrix_view<T>& A,
                                 const const_matrix_view<T>& B,
                        T  beta,             matrix_view<T>& C) const;

        void operator()(ThreadCommunicator& comm,
                        T alpha, const const_matrix_view<T>& A,
                                 const const_matrix_view<T>& B,
                        T  beta,           const_scatter_matrix_view<T>& C) const;

        void operator()(ThreadCommunicator& comm,
                        T alpha,  const const_matrix_view<T>& A,
                                  const const_matrix_view<T>& B,
                        T  beta, block_scatter_matrix<T,MR,NR>& C) const;
    };
};

}
}

#endif
