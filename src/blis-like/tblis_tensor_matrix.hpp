#ifndef _TBLIS_TENSOR_MATRIX_HPP_
#define _TBLIS_TENSOR_MATRIX_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T>
class TensorMatrix
{
    public:
        typedef T type;
        typedef typename real_type<T>::type real_type;

    private:
        trans_op_t _conjtrans;
        type* _ptr;
        dim_t _m;
        dim_t _n;
        dim_t _off_m;
        dim_t _off_n;
        dim_t _m0;
        dim_t _n0;
        inc_t _rs0;
        inc_t _cs0;
        Iterator<> _ri;
        Iterator<> _ci;

    protected:
        void create()
        {
            _conjtrans = BLIS_NO_TRANSPOSE;
            _ptr = NULL;
            _m = 0;
            _n = 0;
            _off_m = 0;
            _off_n = 0;
            _m0 = 0;
            _n0 = 0;
            _rs0 = 0;
            _cs0 = 0;
            _ri.reset();
            _ci.reset();
        }

        void create(const TensorMatrix& other)
        {
            _conjtrans = other._conjtrans;
            _ptr = other._ptr;
            _m = other._m;
            _n = other._n;
            _off_m = other._off_m;
            _off_n = other._off_n;
            _m0 = other._m0;
            _n0 = other._n0;
            _rs0 = other._rs0;
            _cs0 = other._cs0;
            _ri = other._ri;
            _ci = other._ci;
        }

        void create(TensorMatrix&& other)
        {
            _conjtrans = other._conjtrans;
            _ptr = other._ptr;
            _m = other._m;
            _n = other._n;
            _off_m = other._off_m;
            _off_n = other._off_n;
            _m0 = other._m0;
            _n0 = other._n0;
            _rs0 = other._rs0;
            _cs0 = other._cs0;
            _ri = std::move(other._ri);
            _ci = std::move(other._ci);
        }

        void create(const Tensor<T>& other,
                    const std::vector<gint_t>& row_inds,
                    const std::vector<gint_t>& col_inds)
        {
            _conjtrans = BLIS_NO_TRANSPOSE;
            _ptr = other.data();
            _m0 = (row_inds.empty() ? 1 : other.length(row_inds[0]));
            _n0 = (col_inds.empty() ? 1 : other.length(col_inds[0]));
            _rs0 = (row_inds.empty() ? 1 : other.stride(row_inds[0]));
            _cs0 = (col_inds.empty() ? 1 : other.stride(col_inds[0]));
            _m = _m0;
            _n = _n0;
            _off_m = 0;
            _off_n = 0;

            std::vector<dim_t> len_m; len_m.reserve(row_inds.size()-1);
            std::vector<dim_t> len_n; len_n.reserve(col_inds.size()-1);
            std::vector<inc_t> stride_m; stride_m.reserve(row_inds.size()-1);
            std::vector<inc_t> stride_n; stride_n.reserve(col_inds.size()-1);

            for (size_t i = 1;i < row_inds.size();i++)
            {
                _m *= other.length(row_inds[i]);
                len_m.push_back(other.length(row_inds[i]));
                stride_m.push_back(other.stride(row_inds[i]));
            }

            for (size_t i = 1;i < col_inds.size();i++)
            {
                _n *= other.length(col_inds[i]);
                len_n.push_back(other.length(col_inds[i]));
                stride_n.push_back(other.stride(col_inds[i]));
            }

            _ri.reset(len_m, stride_m);
            _ci.reset(len_n, stride_n);
        }

        void create(std::vector<dim_t> len_m,
                    std::vector<dim_t> len_n,
                    T* ptr,
                    std::vector<inc_t> stride_m,
                    std::vector<inc_t> stride_n)
        {
            ASSERT(len_m.size() == stride_m.size());
            ASSERT(len_n.size() == stride_n.size());

            _conjtrans = BLIS_NO_TRANSPOSE;
            _ptr = ptr;
            _m0 = (len_m.empty() ? 1 : len_m[0]);
            _n0 = (len_n.empty() ? 1 : len_n[0]);
            _rs0 = (stride_m.empty() ? 1 : stride_m[0]);
            _cs0 = (stride_n.empty() ? 1 : stride_n[0]);
            _m = _m0;
            _n = _n0;
            _off_m = 0;
            _off_n = 0;

            if (!len_m.empty()) len_m.erase(len_m.begin());
            if (!len_n.empty()) len_n.erase(len_n.begin());
            if (!stride_m.empty()) stride_m.erase(stride_m.begin());
            if (!stride_n.empty()) stride_n.erase(stride_n.begin());

            for (dim_t len : len_m) _m *= len;
            for (dim_t len : len_n) _n *= len;

            _ri.reset(len_m, stride_m);
            _ci.reset(len_n, stride_n);
        }

    public:
        TensorMatrix()
        {
            create();
        }

        TensorMatrix(const TensorMatrix& other)
        {
            create(other);
        }

        TensorMatrix(TensorMatrix&& other)
        {
            create(std::move(other));
        }

        TensorMatrix(const Tensor<T>& other,
                     const std::vector<gint_t>& row_inds,
                     const std::vector<gint_t>& col_inds)
        {
            create(other, row_inds, col_inds);
        }

        TensorMatrix(const std::vector<dim_t>& len_m,
                     const std::vector<dim_t>& len_n,
                     T* ptr,
                     const std::vector<inc_t>& stride_m,
                     const std::vector<inc_t>& stride_n)
        {
            create(len_m, len_n, ptr, stride_m, stride_n);
        }

        TensorMatrix& operator=(const TensorMatrix& other)
        {
            reset(other);
            return *this;
        }

        TensorMatrix& operator=(TensorMatrix&& other)
        {
            reset(std::move(other));
            return *this;
        }

        void reset()
        {
            create();
        }

        void reset(const TensorMatrix& other)
        {
            if (&other == this) return;
            create(other);
        }

        void reset(TensorMatrix&& other)
        {
            if (&other == this) return;
            create(std::move(other));
        }

        void reset(const Tensor<T>& other,
                   const std::vector<gint_t>& row_inds,
                   const std::vector<gint_t>& col_inds)
        {
            create(other, row_inds, col_inds);
        }

        void reset(const std::vector<dim_t>& len_m,
                   const std::vector<dim_t>& len_n,
                   T* ptr,
                   const std::vector<inc_t>& stride_m,
                   const std::vector<inc_t>& stride_n)
        {
            create(len_m, len_n, ptr, stride_m, stride_n);
        }

        bool is_view() const
        {
            return true;
        }

        bool is_transposed() const
        {
            return _conjtrans.transpose();
        }

        bool transpose()
        {
            bool old = is_transposed();
            bli_obj_toggle_trans(_conjtrans);
            return old;
        }

        bool transpose(bool trans)
        {
            bool old = is_transposed();
            bli_obj_set_onlytrans(trans ? BLIS_TRANSPOSE : BLIS_NO_TRANSPOSE, _conjtrans);
            return old;
        }

        trans_t transpose(trans_t trans)
        {
            trans_t old = bli_obj_onlytrans_status(_conjtrans);
            bli_obj_set_onlytrans(trans, _conjtrans);
            return old;
        }

        bool is_conjugated() const
        {
            return _conjtrans.conjugate();
        }

        bool conjugate()
        {
            bool old = is_conjugated();
            bli_obj_toggle_conj(_conjtrans);
            return old;
        }

        bool conjugate(bool conj)
        {
            bool old = is_conjugated();
            bli_obj_set_conj(conj ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE, _conjtrans);
            return old;
        }

        conj_t conjugate(conj_t conj)
        {
            conj_t old = bli_obj_conj_status(_conjtrans);
            bli_obj_set_conj(conj, _conjtrans);
            return old;
        }

        trans_op_t conjtrans()
        {
            return bli_obj_conjtrans_status(_conjtrans);
        }

        trans_op_t conjtrans(trans_op_t conjtrans)
        {
            trans_op_t old = this->conjtrans();
            bli_obj_set_conjtrans(conjtrans, _conjtrans);
            return old;
        }

        dim_t length() const
        {
            return _m;
        }

        dim_t length(dim_t m)
        {
            std::swap(m, _m);
            return m;
        }

        dim_t width() const
        {
            return _n;
        }

        dim_t width(dim_t n)
        {
            std::swap(n, _n);
            return n;
        }

        void shift_down(dim_t m)
        {
            _off_m += m;
        }

        void shift_up(dim_t m)
        {
            shift_down(-m);
        }

        void shift_right(dim_t n)
        {
            _off_n += n;
        }

        void shift_left(dim_t n)
        {
            shift_right(-n);
        }

        void shift_down()
        {
            shift_down(length());
        }

        void shift_up()
        {
            shift_up(length());
        }

        void shift_right()
        {
            shift_right(width());
        }

        void shift_left()
        {
            shift_left(width());
        }

        type* data()
        {
            return _ptr;
        }

        const type* data() const
        {
            return _ptr;
        }

        void row_scatter(inc_t* rscat)
        {
            dim_t p0 = _off_m%_m0;
            inc_t off = p0*_rs0;
            _ri.position(_off_m/_m0, off);

            for (dim_t rscat_idx = 0;_ri.next(off);)
            {
                for (dim_t i0 = p0;i0 < _m0;i0++)
                {
                    if (rscat_idx == _m) return;
                    rscat[rscat_idx++] = off + i0*_rs0;
                }
                p0 = 0;
            }
        }

        void col_scatter(inc_t* cscat)
        {
            dim_t p0 = _off_n%_n0;
            inc_t off = p0*_cs0;
            _ci.position(_off_n/_n0, off);

            for (dim_t cscat_idx = 0;_ci.next(off);)
            {
                for (dim_t j0 = p0;j0 < _n0;j0++)
                {
                    if (cscat_idx == _n) return;
                    cscat[cscat_idx++] = off + j0*_cs0;
                }
                p0 = 0;
            }
        }

        template <dim_t MR>
        void row_block_scatter(inc_t* rs, inc_t* rscat)
        {
            dim_t p0 = _off_m%_m0;
            inc_t off = 0;
            _ri.position(_off_m/_m0, off);

            dim_t nleft = 0;
            for (dim_t rs_idx = 0, rscat_idx = 0;_ri.next(off);)
            {
                for (dim_t i0 = p0;i0 < _m0;i0++)
                {
                    if (rscat_idx == _m) return;

                    if (nleft == 0)
                    {
                        rs[rs_idx++] = (_m0-i0 >= MR || _m0-i0+rscat_idx >= _m ? _rs0 : 0);
                        //rs[rs_idx++] = 0;
                        nleft = MR;
                    }

                    rscat[rscat_idx++] = off + i0*_rs0;
                    nleft--;
                }
                p0 = 0;
            }
        }

        template <dim_t NR>
        void col_block_scatter(inc_t* cs, inc_t* cscat)
        {
            dim_t p0 = _off_n%_n0;
            inc_t off = 0;
            _ci.position(_off_n/_n0, off);

            dim_t nleft = 0;
            for (dim_t cs_idx = 0, cscat_idx = 0;_ci.next(off);)
            {
                for (dim_t j0 = p0;j0 < _n0;j0++)
                {
                    if (cscat_idx == _n) return;

                    if (nleft == 0)
                    {
                        cs[cs_idx++] = (_n0-j0 >= NR || _n0-j0+cscat_idx >= _n ? _cs0 : 0);
                        //cs[cs_idx++] = 0;
                        nleft = NR;
                    }

                    cscat[cscat_idx++] = off + j0*_cs0;
                    nleft--;
                }
                p0 = 0;
            }
        }

        TensorMatrix operator^(trans_op_t trans)
        {
            TensorMatrix view(*this);

            if (trans.transpose()) view.transpose();
            if (trans.conjugate()) view.conjugate();

            return view;
        }

        friend void ViewNoTranspose(TensorMatrix& A, TensorMatrix& V)
        {
            using std::swap;

            blis::detail::AssertNotSelfView(A, V);

            V = A;

            if (V.is_transposed())
            {
                V.transpose(false);
                swap(V._m, V._n);
                swap(V._m0, V._n0);
                swap(V._off_m, V._off_n);
                swap(V._rs0, V._cs0);
                swap(V._ri, V._ci);
            }
        }
};

}
}

#endif
