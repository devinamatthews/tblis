VPATH += $(srcdir)
	
if HAVE_GTEST
bin_PROGRAMS = $(top_builddir)/bin/test
AM_CPPFLAGS = -I./include @gtest_INCLUDES@
__top_builddir__bin_test_LDADD = @gtest_LIBS@
__top_builddir__bin_test_SOURCES = \
	test/algorithm.cxx \
	test/bounded_vector.cxx \
	test/complex.cxx \
	test/cosort.cxx \
	test/global_ptr.cxx \
	test/iostream.cxx \
	test/ptr_list.cxx \
	test/ptr_vector.cxx \
	test/string.cxx \
	test/type_traits.cxx \
	test/vector.cxx \
	test/zip.cxx
endif
