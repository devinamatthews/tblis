#ifndef _TBLIS_CONFIGS_HPP_
#define _TBLIS_CONFIGS_HPP_

#include "util/basic_types.h"
#include "util/thread.h"
#include "util/macros.h"
#include "util/env.hpp"

#include "kernels/1v/add.hpp"
#include "kernels/1v/dot.hpp"
#include "kernels/1v/mult.hpp"
#include "kernels/1v/reduce.hpp"
#include "kernels/1v/scale.hpp"
#include "kernels/1v/set.hpp"
#include "kernels/1v/shift.hpp"

#include "kernels/1f/addf.hpp"
#include "kernels/1f/dotf.hpp"

#include "kernels/1m/trans.hpp"

#include "kernels/3m/gemm.hpp"
#include "kernels/3m/packm.hpp"

namespace tblis
{

//
// Return priority if config can run on this HW, -1 otherwise
//
using check_fn_t = int (*)(void);

template <typename T> struct type_idx;

template <> struct type_idx<   float> { constexpr static int value = 0; };
template <> struct type_idx<  double> { constexpr static int value = 1; };
template <> struct type_idx<scomplex> { constexpr static int value = 2; };
template <> struct type_idx<dcomplex> { constexpr static int value = 3; };

struct blocksize
{
    len_type    _def[4];
    len_type    _max[4];
    len_type   _iota[4];
    len_type _extent[4];

    template <typename T> len_type    def() const { return    _def[type_idx<T>::value]; }
    template <typename T> len_type    max() const { return    _max[type_idx<T>::value]; }
    template <typename T> len_type   iota() const { return   _iota[type_idx<T>::value]; }
    template <typename T> len_type extent() const { return _extent[type_idx<T>::value]; }

    template <template <typename> class BS, typename T> blocksize(const BS<T>&)
    : _def   {BS<float>::def,    BS<double>::def,    BS<scomplex>::def,    BS<dcomplex>::def},
      _max   {BS<float>::max,    BS<double>::max,    BS<scomplex>::max,    BS<dcomplex>::max},
      _iota  {BS<float>::iota,   BS<double>::iota,   BS<scomplex>::iota,   BS<dcomplex>::iota},
      _extent{BS<float>::extent, BS<double>::extent, BS<scomplex>::extent, BS<dcomplex>::extent} {}
};

template <template <typename> class ukr_t>
struct microkernel
{
    void (*_ukr[4])(void);

    template <template <typename> class ukr, typename T> microkernel(const ukr<T>&)
    : _ukr{(void(*)(void))ukr<   float>::value,
           (void(*)(void))ukr<  double>::value,
           (void(*)(void))ukr<scomplex>::value,
           (void(*)(void))ukr<dcomplex>::value} {}

    template <typename T, typename... Args>
    void call(Args&&... args) const
    {
        ((ukr_t<T>)_ukr[type_idx<T>::value])(std::forward<Args>(args)...);
    }
};

template <typename U>
struct parameter
{
    U _val[4];

    template <template <typename> class param, typename T> parameter(const param<T>&)
    : _val{param<   float>::value,
           param<  double>::value,
           param<scomplex>::value,
           param<dcomplex>::value} {}

    template <typename T>
    U value() const
    {
        return _val[type_idx<T>::value];
    }
};

struct config
{
    /*
     * Level 1v kernels
     */

    microkernel<add_ukr_t> add_ukr;
    microkernel<dot_ukr_t> dot_ukr;
    microkernel<mult_ukr_t> mult_ukr;
    microkernel<reduce_ukr_t> reduce_ukr;
    microkernel<scale_ukr_t> scale_ukr;
    microkernel<set_ukr_t> set_ukr;
    microkernel<shift_ukr_t> shift_ukr;

    /*
     * Level 1f kernels
     */

    blocksize addf_nf;
    blocksize dotf_nf;

    microkernel<addf_sum_ukr_t> addf_sum_ukr;
    microkernel<addf_rep_ukr_t> addf_rep_ukr;
    microkernel<dotf_ukr_t> dotf_ukr;

    /*
     * Level 1m kernels
     */

    blocksize trans_mr;
    blocksize trans_nr;

    microkernel<trans_ukr_t> trans_ukr;

    /*
     * GEMM blocksizes and kernels
     */

    blocksize gemm_mr;
    blocksize gemm_nr;
    blocksize gemm_kr;
    blocksize gemm_mc;
    blocksize gemm_nc;
    blocksize gemm_kc;

    microkernel<gemm_ukr_t> gemm_ukr;

    parameter<bool> gemm_row_major;
    parameter<bool> gemm_flip_ukr;

    microkernel<pack_nn_ukr_t> pack_nn_mr_ukr;
    microkernel<pack_nn_ukr_t> pack_nn_nr_ukr;
    microkernel<pack_nnd_ukr_t> pack_nnd_mr_ukr;
    microkernel<pack_nnd_ukr_t> pack_nnd_nr_ukr;
    microkernel<pack_nne_ukr_t> pack_nne_mr_ukr;
    microkernel<pack_nne_ukr_t> pack_nne_nr_ukr;
    microkernel<pack_sn_ukr_t> pack_sn_mr_ukr;
    microkernel<pack_sn_ukr_t> pack_sn_nr_ukr;
    microkernel<pack_ns_ukr_t> pack_ns_mr_ukr;
    microkernel<pack_ns_ukr_t> pack_ns_nr_ukr;
    microkernel<pack_ss_ukr_t> pack_ss_mr_ukr;
    microkernel<pack_ss_ukr_t> pack_ss_nr_ukr;
    microkernel<pack_nb_ukr_t> pack_nb_mr_ukr;
    microkernel<pack_nb_ukr_t> pack_nb_nr_ukr;
    microkernel<pack_sb_ukr_t> pack_sb_mr_ukr;
    microkernel<pack_sb_ukr_t> pack_sb_nr_ukr;
    microkernel<pack_ss_scal_ukr_t> pack_ss_scal_mr_ukr;
    microkernel<pack_ss_scal_ukr_t> pack_ss_scal_nr_ukr;

    parameter<unsigned> m_thread_ratio;
    parameter<unsigned> n_thread_ratio;
    parameter<unsigned> mr_max_thread;
    parameter<unsigned> nr_max_thread;

    check_fn_t check;
    const char* name;

    template <typename Traits> config(const Traits&)
    : add_ukr(typename Traits::template add_ukr<float>()),
      dot_ukr(typename Traits::template dot_ukr<float>()),
      mult_ukr(typename Traits::template mult_ukr<float>()),
      reduce_ukr(typename Traits::template reduce_ukr<float>()),
      scale_ukr(typename Traits::template scale_ukr<float>()),
      set_ukr(typename Traits::template set_ukr<float>()),
      shift_ukr(typename Traits::template shift_ukr<float>()),

      addf_nf(typename Traits::template addf_nf<float>()),
      dotf_nf(typename Traits::template dotf_nf<float>()),

      addf_sum_ukr(typename Traits::template addf_sum_ukr<float>()),
      addf_rep_ukr(typename Traits::template addf_rep_ukr<float>()),
      dotf_ukr(typename Traits::template dotf_ukr<float>()),

      trans_mr(typename Traits::template trans_mr<float>()),
      trans_nr(typename Traits::template trans_nr<float>()),

      trans_ukr(typename Traits::template trans_ukr<float>()),

      gemm_mr(typename Traits::template gemm_mr<float>()),
      gemm_nr(typename Traits::template gemm_nr<float>()),
      gemm_kr(typename Traits::template gemm_kr<float>()),
      gemm_mc(typename Traits::template gemm_mc<float>()),
      gemm_nc(typename Traits::template gemm_nc<float>()),
      gemm_kc(typename Traits::template gemm_kc<float>()),

      gemm_ukr(typename Traits::template gemm_ukr<float>()),

      gemm_row_major(typename Traits::template gemm_row_major<float>()),
      gemm_flip_ukr(typename Traits::template gemm_flip_ukr<float>()),

      pack_nn_mr_ukr(typename Traits::template pack_nn_mr_ukr<float>()),
      pack_nn_nr_ukr(typename Traits::template pack_nn_nr_ukr<float>()),
      pack_nnd_mr_ukr(typename Traits::template pack_nnd_mr_ukr<float>()),
      pack_nnd_nr_ukr(typename Traits::template pack_nnd_nr_ukr<float>()),
      pack_nne_mr_ukr(typename Traits::template pack_nne_mr_ukr<float>()),
      pack_nne_nr_ukr(typename Traits::template pack_nne_nr_ukr<float>()),
      pack_sn_mr_ukr(typename Traits::template pack_sn_mr_ukr<float>()),
      pack_sn_nr_ukr(typename Traits::template pack_sn_nr_ukr<float>()),
      pack_ns_mr_ukr(typename Traits::template pack_ns_mr_ukr<float>()),
      pack_ns_nr_ukr(typename Traits::template pack_ns_nr_ukr<float>()),
      pack_ss_mr_ukr(typename Traits::template pack_ss_mr_ukr<float>()),
      pack_ss_nr_ukr(typename Traits::template pack_ss_nr_ukr<float>()),
      pack_nb_mr_ukr(typename Traits::template pack_nb_mr_ukr<float>()),
      pack_nb_nr_ukr(typename Traits::template pack_nb_nr_ukr<float>()),
      pack_sb_mr_ukr(typename Traits::template pack_sb_mr_ukr<float>()),
      pack_sb_nr_ukr(typename Traits::template pack_sb_nr_ukr<float>()),
      pack_ss_scal_mr_ukr(typename Traits::template pack_ss_scal_mr_ukr<float>()),
      pack_ss_scal_nr_ukr(typename Traits::template pack_ss_scal_nr_ukr<float>()),

      m_thread_ratio(typename Traits::template m_thread_ratio<float>()),
      n_thread_ratio(typename Traits::template n_thread_ratio<float>()),
      mr_max_thread(typename Traits::template mr_max_thread<float>()),
      nr_max_thread(typename Traits::template nr_max_thread<float>()),

      check(Traits::check), name(Traits::name) {}

      operator const tblis_config*() const
      {
          return reinterpret_cast<const tblis_config*>(this);
      }
};

const config& get_default_config();

const config& get_config(const tblis_config* cfg);

const config& get_config(const std::string& name);

}

#endif
