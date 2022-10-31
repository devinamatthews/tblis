VPATH += $(srcdir)

ACLOCAL_AMFLAGS = -I m4

AM_CPPFLAGS = -DMARRAY_ENABLE_ASSERTS -I$(srcdir)/marray -I$(srcdir)/third_party
AM_LDFLAGS = -lpthread

noinst_PROGRAMS = bin/test
bin_test_SOURCES = test/dpd_marray_view.cxx \
                   test/dpd_marray.cxx \
                   test/expression.cxx \
                   test/indexed_dpd_marray.cxx \
                   test/indexed_marray.cxx \
                   test/marray_view.cxx \
                   test/marray.cxx \
                   test/index_iterator.cxx \
                   test/dpd_range.cxx \
                   test/range.cxx \
                   test/short_vector.cxx \
                   third_party/gtest/gtest_main.cc \
                   third_party/gtest/gtest-all.cc

TESTS = bin/test
