#include "tblis.hpp"

#include "util/util.hpp"

namespace tblis
{
namespace blis_like
{

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void MacroKernel<MT,NT>::run<T>::operator()(T alpha, Matrix<T>& A, Matrix<T>& B, T beta,             Matrix<T      >& C) const; \
template void MacroKernel<MT,NT>::run<T>::operator()(T alpha, Matrix<T>& A, Matrix<T>& B, T beta,      ScatterMatrix<T      >& C) const; \
template void MacroKernel<MT,NT>::run<T>::operator()(T alpha, Matrix<T>& A, Matrix<T>& B, T beta, BlockScatterMatrix<T,MR,NR>& C) const;
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

}
}
