lib_LTLIBRARIES = lib/libtblis.la
lib_libtblis_la_SOURCES = \
    src/blis-like/1m/tblis_normfm.cxx \
    \
    src/blis-like/1v/tblis_addv.cxx \
    src/blis-like/1v/tblis_asumv.cxx \
    src/blis-like/1v/tblis_axpbyv.cxx \
    src/blis-like/1v/tblis_axpyv.cxx \
    src/blis-like/1v/tblis_copyv.cxx \
    src/blis-like/1v/tblis_dotv.cxx \
    src/blis-like/1v/tblis_normfv.cxx \
    src/blis-like/1v/tblis_scal2v.cxx \
    src/blis-like/1v/tblis_scalv.cxx \
    src/blis-like/1v/tblis_setv.cxx \
    src/blis-like/1v/tblis_sumv.cxx \
    src/blis-like/1v/tblis_xpbyv.cxx \
    src/blis-like/1v/tblis_zerov.cxx \
    \
    src/blis-like/3/tblis_gemm_ukr.cxx \
    src/blis-like/3/tblis_gemm.cxx \
    src/blis-like/3/tblis_packm.cxx \
    \
	src/core/tensor_c99.cxx \
	\
	src/impl/blas/tensor_contract.cxx \
	src/impl/blas/tensor_mult.cxx \
	src/impl/blas/tensor_outer_prod.cxx \
	src/impl/blas/tensor_sum.cxx \
	src/impl/blas/tensor_weight.cxx \
	src/impl/blis/tensor_contract.cxx \
	\
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
	\
	src/impl/tensor_impl.cxx \
	\
	src/tblis.cxx
	
bin_PROGRAMS = bin/test bin/bench
bin_test_SOURCES = test/test.cxx
bin_bench_SOURCES = test/bench.cxx
	
VPATH += $(srcdir)

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(srcdir) -I$(srcdir)/src @blis_INCLUDES@ @blis___INCLUDES@
bin_test_LDADD = lib/libtblis.la @blis_LIBS@
bin_bench_LDADD = lib/libtblis.la @blis_LIBS@
