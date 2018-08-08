#ifndef _TBLIS_THREAD_HPP_
#define _TBLIS_THREAD_HPP_

#ifdef TBLIS_DONT_USE_CXX11
#ifndef TCI_DONT_USE_CXX11
#define TCI_DONT_USE_CXX11 1
#endif
#endif

#include "tci.h"
#include "basic_types.h"

typedef tci_comm tblis_comm;
extern const tblis_comm* const tblis_single;

#ifdef __cplusplus
extern "C"
{
#endif

unsigned tblis_get_num_threads();

void tblis_set_num_threads(unsigned num_threads);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

#include <vector>
#include <utility>
#include <atomic>
#include <iostream>

namespace tblis
{

using tci::communicator;
using tci::parallelize;
using tci::partition_2x2;

extern communicator single;

extern std::atomic<long> flops;
extern len_type inout_ratio;
extern int outer_threading;

template <typename T, typename=void>
struct atomic_accumulator
{
    std::atomic<T> value;

    constexpr atomic_accumulator(T value = T()) noexcept
    : value(value) {}

    atomic_accumulator& operator=(const atomic_accumulator&) = delete;

    atomic_accumulator& operator=(T val)
    {
        value = val;
        return *this;
    }

    atomic_accumulator& operator+=(T val)
    {
        T old = value.load();
        while (!value.compare_exchange_weak(old, old+val)) continue;
        return *this;
    }

    operator T() const { return value.load(); }
};

template <typename T>
struct atomic_accumulator<T, typename std::enable_if<is_complex<T>::value>::type>
{
        std::atomic<real_type_t<T>> real, imag;

        constexpr atomic_accumulator(T value = T()) noexcept
        : real(value.real()), imag(value.imag()) {}

        atomic_accumulator& operator=(const atomic_accumulator&) = delete;

        atomic_accumulator& operator=(T val)
        {
            real = val.real();
            imag = val.imag();
            return *this;
        }

        atomic_accumulator& operator+=(T val)
        {
            auto old = real.load();
            while (!real.compare_exchange_weak(old, old+val.real())) continue;
            old = imag.load();
            while (!imag.compare_exchange_weak(old, old+val.imag())) continue;
            return *this;
        }

        operator T() const { return {real.load(), imag.load()}; }
};

template <typename T>
struct atomic_reducer_helper
{
    T first;
    len_type second;

    constexpr atomic_reducer_helper(T first = T(), len_type second = 0) noexcept
    : first(first), second(second) {}
};

template <typename T>
using atomic_reducer = std::atomic<atomic_reducer_helper<T>>;

template <typename T>
void atomic_reduce(reduce_t op, atomic_reducer<T>& x, T y_val, len_type y_idx)
{
    auto old = x.load();
    auto update = old;

    do
    {
        update = old;

        switch (op)
        {
            case REDUCE_SUM:
                update.first = old.first + y_val;
                break;
            case REDUCE_SUM_ABS:
                update.first = old.first + std::abs(y_val);
                break;
            case REDUCE_MAX:
                if (y_val > old.first)
                    update = {y_val, y_idx};
                break;
            case REDUCE_MAX_ABS:
                if (std::abs(y_val) > old.first)
                    update = {std::abs(y_val), y_idx};
                break;
            case REDUCE_MIN:
                if (y_val < old.first)
                    update = {y_val, y_idx};
                break;
            case REDUCE_MIN_ABS:
                if (std::abs(y_val) < old.first)
                    update = {std::abs(y_val), y_idx};
                break;
            case REDUCE_NORM_2:
                update.first = old.first + y_val;
                break;
        }
    }
    while (!x.compare_exchange_weak(old, update));
}

template <typename T>
void reduce_init(reduce_t op, T& value, len_type& idx)
{
    typedef std::numeric_limits<real_type_t<T>> limits;

    switch (op)
    {
        case REDUCE_SUM:
        case REDUCE_SUM_ABS:
        case REDUCE_MAX_ABS:
        case REDUCE_NORM_2:
            value = T();
            break;
        case REDUCE_MAX:
            value = limits::lowest();
            break;
        case REDUCE_MIN:
        case REDUCE_MIN_ABS:
            value = limits::max();
            break;
    }

    idx = -1;
}

template <typename T>
atomic_reducer_helper<T> reduce_init(reduce_t op)
{
    T tmp1;
    len_type tmp2;
    reduce_init(op, tmp1, tmp2);
    return {tmp1, tmp2};
}

template <typename T>
void reduce(const communicator& comm, reduce_t op, T& value, len_type& idx)
{
#if TCI_USE_OPENMP_THREADS || TCI_USE_PTHREADS_THREADS || TCI_USE_WINDOWS_THREADS
    if (comm.num_threads() == 1)
    {
#endif

        if (op == REDUCE_NORM_2) value = sqrt(value);
        return;

#if TCI_USE_OPENMP_THREADS || TCI_USE_PTHREADS_THREADS || TCI_USE_WINDOWS_THREADS
    }

    std::vector<std::pair<T,len_type>> vals;
    if (comm.master()) vals.resize(comm.num_threads());

    comm.broadcast(
    [&](std::vector<std::pair<T,len_type>>& vals)
    {
        vals[comm.thread_num()] = {value, idx};
    },
    vals);

    if (comm.master())
    {
        switch (op)
        {
            case REDUCE_SUM:
                for (unsigned i = 1;i < comm.num_threads();i++)
                {
                    vals[0].first += vals[i].first;
                }
                break;
            case REDUCE_SUM_ABS:
                vals[0].first = std::abs(vals[0].first);
                for (unsigned i = 1;i < comm.num_threads();i++)
                {
                    vals[0].first += std::abs(vals[i].first);
                }
                break;
            case REDUCE_MAX:
                for (unsigned i = 1;i < comm.num_threads();i++)
                {
                    if (vals[i].first > vals[0].first) vals[0] = vals[i];
                }
                break;
            case REDUCE_MAX_ABS:
                for (unsigned i = 1;i < comm.num_threads();i++)
                {
                    if (std::abs(vals[i].first) >
                        std::abs(vals[0].first)) vals[0] = vals[i];
                }
                break;
            case REDUCE_MIN:
                for (unsigned i = 1;i < comm.num_threads();i++)
                {
                    if (vals[i].first < vals[0].first) vals[0] = vals[i];
                }
                break;
            case REDUCE_MIN_ABS:
                for (unsigned i = 1;i < comm.num_threads();i++)
                {
                    if (std::abs(vals[i].first) <
                        std::abs(vals[0].first)) vals[0] = vals[i];
                }
                break;
            case REDUCE_NORM_2:
                for (unsigned i = 1;i < comm.num_threads();i++)
                {
                    vals[0].first += vals[i].first;
                }
                vals[0].first = std::sqrt(vals[0].first);
                break;
        }

        value = vals[0].first;
        idx = vals[0].second;
    }

    comm.barrier();
#endif
}

template <typename T>
void reduce(const communicator& comm, T& value)
{
#if TCI_USE_OPENMP_THREADS || TCI_USE_PTHREADS_THREADS || TCI_USE_WINDOWS_THREADS
    if (comm.num_threads() == 1)
    {
#endif

        return;

#if TCI_USE_OPENMP_THREADS || TCI_USE_PTHREADS_THREADS || TCI_USE_WINDOWS_THREADS
    }

    std::vector<T> vals;
    if (comm.master()) vals.resize(comm.num_threads());

    comm.broadcast(
    [&](std::vector<T>& vals)
    {
        vals[comm.thread_num()] = value;
    },
    vals);

    if (comm.master())
    {
        for (unsigned i = 1;i < comm.num_threads();i++)
        {
            vals[0] += vals[i];
        }

        value = vals[0];
    }

    comm.barrier();
#endif
}

template <typename T>
void reduce(const communicator& comm, reduce_t op, atomic_reducer<T>& pair)
{
    T tmp1;
    len_type tmp2;
    tmp1 = pair.load().first;
    tmp2 = pair.load().second;
    reduce(comm, op, tmp1, tmp2);
    pair = {tmp1,tmp2};
}

template <typename T>
void reduce(const communicator& comm, atomic_accumulator<T>& value)
{
    T tmp = value;
    reduce(comm, tmp);
    value = tmp;
}

template <typename Func, typename... Args>
void parallelize_if(const Func& f, const tblis_comm* _comm, Args&&... args)
{
    if (_comm)
    {
        f(*reinterpret_cast<const communicator*>(_comm), args...);
    }
    else
    {
        parallelize
        (
            [&,f](const communicator& comm) mutable
            {
                f(comm, args...);
                comm.barrier();
            },
            tblis_get_num_threads()
        );
    }
}

}

#endif

#endif
