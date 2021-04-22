VPATH += $(srcdir)

ACLOCAL_AMFLAGS = -I m4

AM_CPPFLAGS = -DMARRAY_ENABLE_ASSERTS -I$(srcdir)/include -I$(srcdir)/third_party
AM_LDFLAGS = -lpthread

noinst_PROGRAMS = bin/test
bin_test_SOURCES = test/dpd_marray_view.cxx \
                   test/dpd_marray.cxx \
                   test/dpd_varray_view.cxx \
                   test/dpd_varray.cxx \
                   test/expression.cxx \
                   test/indexed_dpd_varray.cxx \
                   test/indexed_varray.cxx \
                   test/marray_view.cxx \
                   test/marray.cxx \
                   test/miterator.cxx \
                   test/range.cxx \
                   test/short_vector.cxx \
                   test/varray_view.cxx \
                   test/varray.cxx \
                   third_party/gtest/gtest_main.cc \
                   third_party/gtest/gtest-all.cc

TESTS = bin/test
