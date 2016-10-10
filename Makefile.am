lib_LTLIBRARIES = lib/libtblis.la
lib_libtblis_la_SOURCES = \
    src/1m/tblis_normfm.cxx \
    \
    src/1v/tblis_addv.cxx \
    src/1v/tblis_asumv.cxx \
    src/1v/tblis_axpbyv.cxx \
    src/1v/tblis_axpyv.cxx \
    src/1v/tblis_copyv.cxx \
    src/1v/tblis_dotv.cxx \
    src/1v/tblis_normfv.cxx \
    src/1v/tblis_scal2v.cxx \
    src/1v/tblis_scalv.cxx \
    src/1v/tblis_setv.cxx \
    src/1v/tblis_sumv.cxx \
    src/1v/tblis_xpbyv.cxx \
    src/1v/tblis_zerov.cxx \
    \
    src/3/tblis_gemm.cxx \
	\
	src/impl/blas/tensor_contract.cxx \
	src/impl/blas/tensor_mult.cxx \
	src/impl/blas/tensor_outer_prod.cxx \
	src/impl/blas/tensor_weight.cxx \
	\
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
	src/tblis_matrify.cxx \
	src/tblis_mutex.cxx
	
noinst_LTLIBRARIES =
lib_libtblis_la_LIBADD =
	
#if ENABLE_ARMV7A
#noinst_LTLIBRARIES += lib/libarmv7a.la
#lib_libtblis_la_LIBADD += lib/libarmv7a.la
#lib_libarmv7a_la_SOURCES = src/external/blis/config/armv7a/kernels/3/bli_cgemm_kernel_2x2.S \
#                           src/external/blis/config/armv7a/kernels/3/bli_dgemm_kernel_4x4.S \
#                           src/external/blis/config/armv7a/kernels/3/bli_sgemm_kernel_4x4.S \
#                           src/external/blis/config/armv7a/kernels/3/bli_zgemm_kernel_2x2.S \
#                           src/external/blis/config/armv7a/kernels/3/bli_gemm_opt_4x4.c
#lib_libarmv7a_la_CCASFLAGS = -mfloat-abi=hard -mfpu=vfpv3 -marm -march=armv7-a
#lib_libarmv7a_la_CFLAGS = -Isrc/external/blis/config/armv7a -mfloat-abi=hard -mfpu=vfpv3 -marm -march=armv7-a
#endif
#
#if ENABLE_ARMV8A
#noinst_LTLIBRARIES += lib/libarmv8a.la
#lib_libtblis_la_LIBADD += lib/libarmv8a.la
#lib_libarmv8a_la_SOURCES = src/external/blis/config/armv8a/kernels/3/bli_gemm_opt_4x4.c
#lib_libarmv8a_la_CFLAGS = -Isrc/external/blis/config/armv8a -march=armv8-a+fp+simd -mcpu=cortex-a57.cortex-a53
#endif
#
#if ENABLE_BGQ
#noinst_LTLIBRARIES += lib/libbgq.la
#lib_libtblis_la_LIBADD += lib/libbgq.la
#lib_libbgq_la_SOURCES = src/external/blis/config/bgq/kernels/3/bli_gemm_int_8x8.c
#lib_libbgq_la_CFLAGS = -Isrc/external/blis/config/bgq -qasm=gcc -qkeyword=asm -qprefetch -qunroll=yes
#endif
#
#if ENABLE_BULLDOZER
#noinst_LTLIBRARIES += lib/libbulldozer.la
#lib_libtblis_la_LIBADD += lib/libbulldozer.la
#lib_libbulldozer_la_SOURCES = src/external/blis/config/bulldozer/kernels/3/bli_gemm_asm_d4x6_fma4.c
#lib_libbulldozer_la_CFLAGS = -Isrc/external/blis/config/bulldozer -mavx -mfma -march=bdver2 -mfpmath=sse
#endif
#
#if ENABLE_CARRIZO
#noinst_LTLIBRARIES += lib/libcarrizo.la
#lib_libtblis_la_LIBADD += lib/libcarrizo.la
#lib_libcarrizo_la_SOURCES = src/external/blis/config/carrizo/kernels/3/bli_gemm_asm_d8x3.c
#lib_libcarrizo_la_CFLAGS = -Isrc/external/blis/config/carrizo -mavx -mfma -march=bdver4 -mfpmath=sse
#endif
#
#if ENABLE_CORTEX_A15
#noinst_LTLIBRARIES += lib/libcortex-a15.la
#lib_libtblis_la_LIBADD += lib/libcortex-a15.la
#lib_libcortex_a15_la_SOURCES = src/external/blis/config/cortex-a15/kernels/3/bli_gemm_opt_4x4.c
#lib_libcortex_a15_la_CFLAGS = -Isrc/external/blis/config/cortex-a15 -mfloat-abi=hard -mfpu=neon -march=armv7-a
#endif
#
#if ENABLE_CORTEX_A9
#noinst_LTLIBRARIES += lib/libcortex-a9.la
#lib_libtblis_la_LIBADD += lib/libcortex-a9.la
#lib_libcortex_a9_la_SOURCES = src/external/blis/config/cortex-a9/kernels/3/bli_gemm_opt_4x4.c
#lib_libcortex_a9_la_CFLAGS = -Isrc/external/blis/config/cortex-a9 -mfloat-abi=hard -mfpu=neon -march=armv7-a
#endif
#
#if ENABLE_DUNNINGTON
#noinst_LTLIBRARIES += lib/libdunnington.la
#lib_libtblis_la_LIBADD += lib/libdunnington.la
#lib_libdunnington_la_SOURCES = src/external/blis/config/dunnington/kernels/3/bli_gemm_asm_d4x4.c
#if ENABLE_INTEL_COMPILER
#lib_libdunnington_la_CFLAGS = -Isrc/external/blis/config/dunnington -xSSE4.2
#else
#lib_libdunnington_la_CFLAGS = -Isrc/external/blis/config/dunnington -msse3 -march=corei7 -mfpmath=sse
#endif
#endif
#
#if ENABLE_HASWELL
#noinst_LTLIBRARIES += lib/libhaswell.la
#lib_libtblis_la_LIBADD += lib/libhaswell.la
#lib_libhaswell_la_SOURCES = src/external/blis/config/haswell/kernels/3/bli_gemm_asm_d12x4.c \
#						    src/external/blis/config/haswell/kernels/3/bli_gemm_asm_d8x6.c \
#						    src/external/blis/config/haswell/kernels/3/bli_gemm_asm_d6x8.c \
#						    src/external/blis/config/haswell/kernels/3/bli_gemm_asm_d4x12.c
#if ENABLE_INTEL_COMPILER
#lib_libhaswell_la_CFLAGS = -Isrc/external/blis/config/haswell -xCORE-AVX2
#else
#lib_libhaswell_la_CFLAGS = -Isrc/external/blis/config/haswell -mavx2 -mfma -mfpmath=sse -march=core-avx2
#endif
#endif
#
#if ENABLE_KNL
#noinst_LTLIBRARIES += lib/libknl.la
#lib_libtblis_la_LIBADD += lib/libknl.la
#lib_libknl_la_SOURCES = src/external/blis/config/knl/kernels/1m/bli_packm_opt_24x8.c \
#					    src/external/blis/config/knl/kernels/3/bli_dgemm_opt_12x16.c \
#					    src/external/blis/config/knl/kernels/3/bli_dgemm_opt_24x8.c \
#					    src/external/blis/config/knl/kernels/3/bli_dgemm_opt_8x24.c \
#					    src/external/blis/config/knl/kernels/3/bli_dgemm_opt_30x8_att.c \
#					    src/external/blis/config/knl/kernels/3/bli_sgemm_opt_30x16_att.c
#if ENABLE_INTEL_COMPILER
#lib_libknl_la_CFLAGS = -Isrc/external/blis/config/knl -xMIC-AVX512
#else
#if IS_OSX
#lib_libknl_la_CFLAGS = -Isrc/external/blis/config/knl -mavx512f -mavx512pf -mfpmath=sse -march=knl -Wa,-march=knl
#else
#lib_libknl_la_CFLAGS = -Isrc/external/blis/config/knl -mavx512f -mavx512pf -mfpmath=sse -march=knl
#endif
#endif
#endif
#
#if ENABLE_LOONGSON3A
#noinst_LTLIBRARIES += lib/libloongson3a.la
#lib_libtblis_la_LIBADD += lib/libloongson3a.la
#lib_libloongson3a_la_SOURCES = src/external/blis/config/loongson3a/kernels/3/bli_gemm_opt_d4x4.c
#lib_libloongson3a_la_CFLAGS = -Isrc/external/blis/config/loongson3a -march=loongson3a -mtune=loongson3a
#endif
#
#if ENABLE_MIC
#noinst_LTLIBRARIES += lib/libmic.la
#lib_libtblis_la_LIBADD += lib/libmic.la
#lib_libmic_la_SOURCES = src/external/blis/config/mic/kernels/3/bli_dgemm_opt_30x8.c \
#					    src/external/blis/config/mic/kernels/3/bli_sgemm_opt_30x16.c
#lib_libmic_la_CFLAGS = -Isrc/external/blis/config/mic -fasm-blocks
#endif
#
#if ENABLE_PILEDRIVER
#noinst_LTLIBRARIES += lib/libpiledriver.la
#lib_libtblis_la_LIBADD += lib/libpiledriver.la
#lib_libpiledriver_la_SOURCES = src/external/blis/config/piledriver/kernels/3/bli_gemm_asm_d8x3.c
#lib_libpiledriver_la_CFLAGS = -Isrc/external/blis/config/piledriver -mavx -mfma -march=bdver2 -mfpmath=sse
#endif
#
#if ENABLE_POWER7
#noinst_LTLIBRARIES += lib/libpower7.la
#lib_libtblis_la_LIBADD += lib/libpower7.la
#lib_libpower7_la_SOURCES = src/external/blis/config/power7/kernels/3/bli_gemm_opt_8x4.c
#lib_libpower7_la_CFLAGS = -Isrc/external/blis/config/power7 -mcpu=power7 -mtune=power7 -mvsx
#endif
#
#if ENABLE_SANDYBRIDGE
#noinst_LTLIBRARIES += lib/libsandybridge.la
#lib_libtblis_la_LIBADD += lib/libsandybridge.la
#lib_libsandybridge_la_SOURCES = src/external/blis/config/sandybridge/kernels/3/bli_gemm_asm_d8x4.c
#if ENABLE_INTEL_COMPILER
#lib_libsandybridge_la_CFLAGS = -Isrc/external/blis/config/sandybridge -xAVX
#else
#lib_libsandybridge_la_CFLAGS = -Isrc/external/blis/config/sandybridge -mavx -mfpmath=sse -march=corei7-avx
#endif
#endif

bin_PROGRAMS = bin/test
if ENABLE_BLAS
bin_PROGRAMS += bin/bench
endif
bin_test_SOURCES = test/test.cxx
bin_bench_SOURCES = test/bench.cxx

VPATH += $(srcdir)

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(srcdir) -I$(srcdir)/src -I. -Isrc
AM_LDFLAGS = -pthread
bin_test_LDADD = lib/libtblis.la
bin_bench_LDADD = lib/libtblis.la $(BLAS_LIBS)
