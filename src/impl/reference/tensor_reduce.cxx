#include "impl/tensor_impl.hpp"
#include "util/iterator.hpp"

using namespace std;
using namespace blis;
using namespace tensor::util;

namespace tensor
{
namespace impl
{

template <typename T>
int tensor_reduce_reference(reduce_t op, const Tensor<T>& A, const std::string& idx_A, T& val, inc_t& idx)
{
    Iterator iter_A(A.getLengths(), A.getStrides());

    const T* restrict A_  = A.getData();
    const T* const    A0_ = A_;

    switch (op)
    {
        case REDUCE_SUM:
        case REDUCE_SUM_ABS:
        case REDUCE_NORM_2:
            val = 0.0;
            break;
        case REDUCE_MAX:
        case REDUCE_MAX_ABS:
            val = numeric_limits<typename real_type<T>::type>::lowest();
            break;
        case REDUCE_MIN:
        case REDUCE_MIN_ABS:
            val = numeric_limits<typename real_type<T>::type>::max();
            break;
    }

    idx = 0;

    switch (op)
    {
        case REDUCE_SUM:
            for (;iter_A.nextIteration(A_);)
            {
                assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
                val += *A_;
            }
            break;
        case REDUCE_SUM_ABS:
            for (;iter_A.nextIteration(A_);)
            {
                assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
                val += std::abs(*A_);
            }
            break;
        case REDUCE_MAX:
            for (;iter_A.nextIteration(A_);)
            {
                assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
                if (*A_ > val)
                {
                    val = *A_;
                    idx = A_-A0_;
                }
            }
            break;
        case REDUCE_MAX_ABS:
            for (;iter_A.nextIteration(A_);)
            {
                assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
                if (std::abs(*A_) > val)
                {
                    val = std::abs(*A_);
                    idx = A_-A0_;
                }
            }
            break;
        case REDUCE_MIN:
            for (;iter_A.nextIteration(A_);)
            {
                assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
                if (*A_ < val)
                {
                    val = *A_;
                    idx = A_-A0_;
                }
            }
            break;
        case REDUCE_MIN_ABS:
            for (;iter_A.nextIteration(A_);)
            {
                assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
                if (std::abs(*A_) < val)
                {
                    val = std::abs(*A_);
                    idx = A_-A0_;
                }
            }
            break;
        case REDUCE_NORM_2:
            for (;iter_A.nextIteration(A_);)
            {
                assert (A_-A.getData() >= 0 && A_-A.getData() < A.getDataSize());
                val += norm2(*A_);
            }
            val = sqrt(real_part(val));
            break;
    }

    return 0;
}

template
int tensor_reduce_reference<   float>(reduce_t op, const Tensor<   float>& A, const std::string& idx_A,    float& val, inc_t& idx);

template
int tensor_reduce_reference<  double>(reduce_t op, const Tensor<  double>& A, const std::string& idx_A,   double& val, inc_t& idx);

template
int tensor_reduce_reference<sComplex>(reduce_t op, const Tensor<sComplex>& A, const std::string& idx_A, sComplex& val, inc_t& idx);

template
int tensor_reduce_reference<dComplex>(reduce_t op, const Tensor<dComplex>& A, const std::string& idx_A, dComplex& val, inc_t& idx);

}
}
