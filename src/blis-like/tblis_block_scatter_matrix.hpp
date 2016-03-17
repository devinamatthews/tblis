#ifndef _TBLIS_BLOCK_SCATTER_MATRIX_HPP_
#define _TBLIS_BLOCK_SCATTER_MATRIX_HPP_

#include "tblis.hpp"

#include "util/util.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T, dim_t MB_, dim_t NB_>
class BlockScatterMatrix
{
    public:
        typedef T type;
        typedef typename real_type<T>::type real_type;

    private:
        trans_op_t _conjtrans;
        type* _ptr;
        dim_t _m;
        dim_t _n;
        dim_t _mb;
        dim_t _nb;
        inc_t* _rs;
        inc_t* _cs;
        inc_t* _rscat;
        inc_t* _cscat;
        dim_t _m_sub;
        dim_t _n_sub;
        inc_t _m_off;
        inc_t _n_off;
        inc_t _ptr_off;

        constexpr static bool M_BLOCKED = (MB_ != 0);
        constexpr static bool N_BLOCKED = (NB_ != 0);
        constexpr static dim_t MB = (MB_ == 0 ? 1 : MB_);
        constexpr static dim_t NB = (NB_ == 0 ? 1 : NB_);

    protected:
        void create()
        {
            _conjtrans = BLIS_NO_TRANSPOSE;
            _ptr = NULL;
            _m = 0;
            _n = 0;
            _mb = 0;
            _nb = 0;
            _rs = NULL;
            _cs = NULL;
            _rscat = NULL;
            _cscat = NULL;
            _m_sub = 0;
            _n_sub = 0;
            _m_off = 0;
            _n_off = 0;
            _ptr_off = 0;
        }

        void create(dim_t m, dim_t n, type* p, inc_t* rs, inc_t* cs,
                    inc_t* rscat, inc_t* cscat)
        {
            _conjtrans = BLIS_NO_TRANSPOSE;
            _ptr = p;
            _m = m;
            _n = n;
            _mb = (m+MB-1)/MB;
            _nb = (n+NB-1)/NB;
            _rs = rs;
            _cs = cs;
            _rscat = rscat;
            _cscat = cscat;
            _m_sub = 0;
            _n_sub = 0;
            _m_off = 0;
            _n_off = 0;
        }

    public:
        BlockScatterMatrix()
        {
            create();
        }

        BlockScatterMatrix(const BlockScatterMatrix&) = default;

        BlockScatterMatrix(dim_t m, dim_t n, type* p, inc_t* rs, inc_t* cs,
                           inc_t* rscat, inc_t* cscat)
        {
            create(m, n, p, rs, cs, rscat, cscat);
        }

        BlockScatterMatrix& operator=(const BlockScatterMatrix&) = default;

        void reset()
        {
            create();
        }

        void reset(const BlockScatterMatrix& other)
        {
            *this = other;
        }

        void reset(dim_t m, dim_t n, type* p, inc_t* rs, inc_t* cs,
                   inc_t* rscat, inc_t* cscat)
        {
            create(m, n, p, rs, cs, rscat, cscat);
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
            return _m_sub;
        }

        dim_t length(dim_t m)
        {
            std::swap(m, _m_sub);
            return m;
        }

        dim_t width() const
        {
            return _n_sub;
        }

        dim_t width(dim_t n)
        {
            std::swap(n, _n_sub);
            return n;
        }

        inc_t row_stride() const
        {
            return (M_BLOCKED ? _rs[_m_off] : 0);
        }

        inc_t col_stride() const
        {
            return (N_BLOCKED ? _cs[_n_off] : 0);
        }

        const inc_t* row_scatter() const
        {
            static inc_t end = 0;
            return (_m_off == _mb ? &end : _rscat + _m_off*MB);
        }

        const inc_t* col_scatter() const
        {
            static inc_t end = 0;
            return (_n_off == _nb ? &end : _cscat + _n_off*NB);
        }

        void shift_down(dim_t m)
        {
            _ptr_off -= *row_scatter();
            if (m < 0)
            {
                _m_off += (m-MB+1)/MB;
            }
            else
            {
                _m_off += (m+MB-1)/MB;
            }
            _ptr_off += *row_scatter();
        }

        void shift_up(dim_t m)
        {
            shift_down(-m);
        }

        void shift_right(dim_t n)
        {
            _ptr_off -= *col_scatter();
            if (n < 0)
            {
                _n_off += (n-NB+1)/NB;
            }
            else
            {
                _n_off += (n+NB-1)/NB;
            }
            _ptr_off += *col_scatter();
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
            return _ptr + (row_stride() == 0 ? 0 : *row_scatter())
                        + (col_stride() == 0 ? 0 : *col_scatter());
        }

        const type* data() const
        {
            return _ptr + (row_stride() == 0 ? 0 : *row_scatter())
                        + (col_stride() == 0 ? 0 : *col_scatter());
        }

        BlockScatterMatrix operator^(trans_op_t trans)
        {
            BlockScatterMatrix view(*this);

            if (trans.transpose()) view.transpose();
            if (trans.conjugate()) view.conjugate();

            return view;
        }

        friend void ViewNoTranspose(BlockScatterMatrix& A, BlockScatterMatrix& V)
        {
            blis::detail::AssertNotSelfView(A, V);

            if (A.is_conjugated()) V.conjugate();

            if (A.is_transposed())
            {
                V.reset(A._n, A._m, A._ptr, A._cs, A._rs, A._cscat, A._rscat);
            }
            else
            {
                V.reset(A._m, A._m, A._ptr, A._rs, A._cs, A._rscat, A._cscat);
            }
        }
};

}
}

#endif
