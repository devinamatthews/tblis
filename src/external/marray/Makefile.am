VPATH += $(srcdir)

ACLOCAL_AMFLAGS = -I m4

AM_CPPFLAGS = -DMARRAY_ENABLE_ASSERTS -I$(srcdir)/include -Ithird_party/googletest/include

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
                   test/utility.cxx \
                   test/varray_view.cxx \
                   test/varray.cxx \
                   test/viterator.cxx
bin_test_LDADD = third_party/googletest/lib/libgtest_main.la \
                 third_party/googletest/lib/libgtest.la

TESTS = bin/test

SUBDIRS = third_party/googletest
