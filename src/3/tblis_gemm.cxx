#include "../util/tblis_util.hpp"
#include "tblis.hpp"


namespace tblis
{
namespace blis_like
{

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void MacroKernel<MT,NT>::run<T>::operator()(ThreadCommunicator& comm, T alpha, Matrix<T>& A, Matrix<T>& B, T beta,             Matrix<T      >& C); \
template void MacroKernel<MT,NT>::run<T>::operator()(ThreadCommunicator& comm, T alpha, Matrix<T>& A, Matrix<T>& B, T beta,      ScatterMatrix<T      >& C); \
template void MacroKernel<MT,NT>::run<T>::operator()(ThreadCommunicator& comm, T alpha, Matrix<T>& A, Matrix<T>& B, T beta, BlockScatterMatrix<T,MR,NR>& C);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

}
}
