bin_PROGRAMS = $(top_builddir)/bin/test $(top_builddir)/bin/bench
tensor_SOURCES = \
	src/core/tensor_c99.cxx \
	src/impl/blas/tensor_contract.cxx \
	src/impl/blas/tensor_mult.cxx \
	src/impl/blas/tensor_outer_prod.cxx \
	src/impl/blas/tensor_sum.cxx \
	src/impl/blas/tensor_weight.cxx \
	src/impl/blis/tensor_contract.cxx \
	src/impl/reference/tensor_contract.cxx \
	src/impl/reference/tensor_dot.cxx \
	src/impl/reference/tensor_mult.cxx \
	src/impl/reference/tensor_outer_prod.cxx \
	src/impl/reference/tensor_reduce.cxx \
	src/impl/reference/tensor_replicate.cxx \
	src/impl/reference/tensor_scale.cxx \
	src/impl/reference/tensor_sum.cxx \
	src/impl/reference/tensor_trace.cxx \
	src/impl/reference/tensor_transpose.cxx \
	src/impl/reference/tensor_weight.cxx \
	src/impl/tensor_impl.cxx
__top_builddir__bin_test_SOURCES = $(tensor_SOURCES) src/test.cxx
__top_builddir__bin_bench_SOURCES = $(tensor_SOURCES) src/bench.cxx
	
VPATH += $(srcdir)

AM_CPPFLAGS = -I$(srcdir)/src @blis_INCLUDES@
__top_builddir__bin_test_LDADD = @blis_LIBS@
__top_builddir__bin_bench_LDADD = @blis_LIBS@
