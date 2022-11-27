#ifndef TBLIS_INTERNAL_THREAD_HPP
#define TBLIS_INTERNAL_THREAD_HPP 1

#include <vector>
#include <utility>
#include <atomic>
#include <iostream>

#include <tblis/base/thread.h>

#include <tblis/internal/types.hpp>

TBLIS_BEGIN_NAMESPACE

extern std::atomic<long> flops;
extern len_type inout_ratio;
extern int outer_threading;

struct atomic_accumulator
{
    std::atomic<float> s;
    std::atomic<double> d;
    std::atomic<float> cr, ci;
    std::atomic<double> zr, zi;

    template <typename T>
    static void accumulate(std::atomic<T>& value, T val)
    {
        T old = value.load();
        while (!value.compare_exchange_weak(old, old+val)) continue;
    }

    atomic_accumulator() noexcept
    : s(0.0f), d(0.0), cr(0.0f), ci(0.0f), zr(0.0), zi(0.0) {}

    atomic_accumulator(const tblis_scalar& val) noexcept
    {
        *this = val;
    }

    atomic_accumulator& operator=(const atomic_accumulator&) = delete;

    atomic_accumulator& operator=(const tblis_scalar& val)
    {
        if (val.type & DOUBLE)
        {
            zr = val.data.z.real();
            zi = val.data.z.imag();
        }
        else
        {
            cr = val.data.c.real();
            ci = val.data.c.imag();
        }

        return *this;
    }

    atomic_accumulator& operator+=(const tblis_scalar& val)
    {
        switch (val.type)
        {
            case FLOAT:    accumulate(s, val.data.s); break;
            case DOUBLE:   accumulate(d, val.data.d); break;
            case SCOMPLEX: accumulate(cr, val.data.c.real());
                           accumulate(ci, val.data.c.imag()); break;
            case DCOMPLEX: accumulate(zr, val.data.z.real());
                           accumulate(zi, val.data.z.imag()); break;

        }
        return *this;
    }

    void store(type_t type, char* result) const
    {
        switch (type)
        {
            case FLOAT:
                {
                    float tmp = s.load();
                    memcpy(result, &tmp, sizeof(tmp));
                }
                break;
            case DOUBLE:
                {
                    double tmp = d.load();
                    memcpy(result, &tmp, sizeof(tmp));
                }
                break;
            case SCOMPLEX:
                {
                    scomplex tmp{cr.load(), ci.load()};
                    memcpy(result, &tmp, sizeof(tmp));
                }
                break;
            case DCOMPLEX:
                {
                    dcomplex tmp{zr.load(), zi.load()};
                    memcpy(result, &tmp, sizeof(tmp));
                }
                break;
        }
    }
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

inline void reduce_init(reduce_t op, tblis_scalar& value, len_type& idx)
{
    switch (value.type)
    {
        case FLOAT:    reduce_init(op, value.data.s, idx); break;
        case DOUBLE:   reduce_init(op, value.data.d, idx); break;
        case SCOMPLEX: reduce_init(op, value.data.c, idx); break;
        case DCOMPLEX: reduce_init(op, value.data.z, idx); break;
    }
}

template <typename T>
atomic_reducer_helper<T> reduce_init(reduce_t op)
{
    T tmp1;
    len_type tmp2;
    reduce_init(op, tmp1, tmp2);
    return {tmp1, tmp2};
}

struct atomic_reducer
{
    std::atomic<atomic_reducer_helper<float>> s;
    std::atomic<atomic_reducer_helper<double>> d;
    std::atomic<atomic_reducer_helper<scomplex>> c;
    std::atomic<atomic_reducer_helper<dcomplex>> z;

    atomic_reducer(reduce_t op)
    : s(reduce_init<float>(op)),
      d(reduce_init<double>(op)),
      c(reduce_init<scomplex>(op)),
      z(reduce_init<dcomplex>(op)) {}

    void store(type_t type, char* val, len_type& idx)
    {
        switch (type)
        {
            case FLOAT:
                {
                    float tmp = s.load().first;
                    memcpy(val, &tmp, sizeof(tmp));
                    idx = s.load().second;
                }
                break;
            case DOUBLE:
                {
                    double tmp = d.load().first;
                    memcpy(val, &tmp, sizeof(tmp));
                    idx = d.load().second;
                }
                break;
            case SCOMPLEX:
                {
                    scomplex tmp = c.load().first;
                    memcpy(val, &tmp, sizeof(tmp));
                    idx = c.load().second;
                }
                break;
            case DCOMPLEX:
                {
                    dcomplex tmp = z.load().first;
                    memcpy(val, &tmp, sizeof(tmp));
                    idx = z.load().second;
                }
                break;
        }
    }
};

template <typename T>
void atomic_reduce(reduce_t op, std::atomic<atomic_reducer_helper<T>>& x, T y_val, len_type y_idx)
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
                if (old.first < y_val)
                    update = {y_val, y_idx};
                break;
            case REDUCE_MAX_ABS:
                if (old.first < std::abs(y_val))
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

inline void atomic_reduce(reduce_t op, atomic_reducer& x,
                          const tblis_scalar& y_val, len_type y_idx)
{
    switch (y_val.type)
    {
        case FLOAT:    atomic_reduce(op, x.s, y_val.data.s, y_idx); break;
        case DOUBLE:   atomic_reduce(op, x.d, y_val.data.d, y_idx); break;
        case SCOMPLEX: atomic_reduce(op, x.c, y_val.data.c, y_idx); break;
        case DCOMPLEX: atomic_reduce(op, x.z, y_val.data.z, y_idx); break;
    }
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
                    if (vals[0].first < vals[i].first) vals[0] = vals[i];
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
void reduce(const communicator& comm, reduce_t op, std::atomic<atomic_reducer_helper<T>>& pair)
{
    T tmp1;
    len_type tmp2;
    tmp1 = pair.load().first;
    tmp2 = pair.load().second;
    reduce(comm, op, tmp1, tmp2);
    pair = {tmp1,tmp2};
}

inline void reduce(type_t type, const communicator& comm, reduce_t op, atomic_reducer& pair)
{
    switch (type)
    {
        case FLOAT:    reduce(comm, op, pair.s); break;
        case DOUBLE:   reduce(comm, op, pair.d); break;
        case SCOMPLEX: reduce(comm, op, pair.c); break;
        case DCOMPLEX: reduce(comm, op, pair.z); break;
    }
}

inline void reduce(type_t type, const communicator& comm, atomic_accumulator& value)
{
    switch (type)
    {
        case FLOAT:
            {
                float tmp = value.s.load();
                reduce(comm, tmp);
                value.s = tmp;
            }
            break;
        case DOUBLE:
            {
                double tmp = value.d.load();
                reduce(comm, tmp);
                value.d = tmp;
            }
            break;
        case SCOMPLEX:
            {
                scomplex tmp(value.cr.load(), value.ci.load());
                reduce(comm, tmp);
                value.cr = tmp.real();
                value.ci = tmp.imag();
            }
            break;
        case DCOMPLEX:
            {
                dcomplex tmp(value.zr.load(), value.zi.load());
                reduce(comm, tmp);
                value.zr = tmp.real();
                value.zi = tmp.imag();
            }
            break;
    }
}

TBLIS_END_NAMESPACE

#endif //TBLIS_INTERNAL_THREAD_HPP
