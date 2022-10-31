#include "gtest/gtest.h"

#include "complex.hpp"

using namespace std;
using namespace stl_ext;

TEST(unit_complex, traits)
{
    EXPECT_TRUE((is_same<real_type_t<double>,double>::value));
    EXPECT_TRUE((is_same<real_type_t<complex<double>>,double>::value));
    EXPECT_TRUE((is_same<complex_type_t<double>,complex<double>>::value));
    EXPECT_TRUE((is_same<complex_type_t<complex<double>>,complex<double>>::value));
    EXPECT_TRUE((is_complex<complex<double>>::value));
    EXPECT_FALSE((is_complex<double>::value));
}

#define COMPLEX_OP_REAL_2(op, type1, type2, real_in, imag_in, other, real_out, imag_out) \
EXPECT_EQ((complex<common_type_t<type1,type2>>(real_out, imag_out)), \
          (complex<type1>(real_in, imag_in) op (type2)(other)));

#define COMPLEX_OP_REAL_1(op, type1, real_in, imag_in, other, real_out, imag_out) \
COMPLEX_OP_REAL_2(op, type1, float, real_in, imag_in, other, real_out, imag_out); \
COMPLEX_OP_REAL_2(op, type1, double, real_in, imag_in, other, real_out, imag_out); \
COMPLEX_OP_REAL_2(op, type1, long double, real_in, imag_in, other, real_out, imag_out);

#define COMPLEX_OP_REAL(op, real_in, imag_in, other, real_out, imag_out) \
COMPLEX_OP_REAL_1(op, float, real_in, imag_in, other, real_out, imag_out); \
COMPLEX_OP_REAL_1(op, double, real_in, imag_in, other, real_out, imag_out); \
COMPLEX_OP_REAL_1(op, long double, real_in, imag_in, other, real_out, imag_out);

#define REAL_OP_COMPLEX_2(op, type1, type2, real_in, imag_in, other, real_out, imag_out) \
EXPECT_EQ((complex<common_type_t<type1,type2>>(real_out, imag_out)), \
          ((type2)(other) op complex<type1>(real_in, imag_in)));

#define REAL_OP_COMPLEX_1(op, type1, real_in, imag_in, other, real_out, imag_out) \
REAL_OP_COMPLEX_2(op, type1, float, real_in, imag_in, other, real_out, imag_out); \
REAL_OP_COMPLEX_2(op, type1, double, real_in, imag_in, other, real_out, imag_out); \
REAL_OP_COMPLEX_2(op, type1, long double, real_in, imag_in, other, real_out, imag_out);

#define REAL_OP_COMPLEX(op, real_in, imag_in, other, real_out, imag_out) \
REAL_OP_COMPLEX_1(op, float, real_in, imag_in, other, real_out, imag_out); \
REAL_OP_COMPLEX_1(op, double, real_in, imag_in, other, real_out, imag_out); \
REAL_OP_COMPLEX_1(op, long double, real_in, imag_in, other, real_out, imag_out);

TEST(unit_complex, operators)
{
    COMPLEX_OP_REAL(+, 1.0, 1.0, 2.0,  3.0,  1.0);
    REAL_OP_COMPLEX(+, 1.0, 1.0, 2.0,  3.0,  1.0);
    COMPLEX_OP_REAL(-, 1.0, 1.0, 2.0, -1.0,  1.0);
    REAL_OP_COMPLEX(-, 1.0, 1.0, 2.0,  1.0, -1.0);
    COMPLEX_OP_REAL(*, 1.0, 1.0, 2.0,  2.0,  2.0);
    REAL_OP_COMPLEX(*, 1.0, 1.0, 2.0,  2.0,  2.0);
    COMPLEX_OP_REAL(/, 1.0, 1.0, 2.0,  0.5,  0.5);
    REAL_OP_COMPLEX(/, 1.0, 1.0, 2.0,  1.0, -1.0);
}
