lib_LTLIBRARIES = lib/libtblis.la
lib_libtblis_la_SOURCES = \
    \
    tblis/iface/1t/add.cxx \
    tblis/iface/1t/dot.cxx \
    tblis/iface/1t/reduce.cxx \
    tblis/iface/1t/scale.cxx \
    tblis/iface/1t/set.cxx \
    tblis/iface/1t/shift.cxx \
    \
    tblis/iface/3t/mult.cxx \
    \
    tblis/internal/0/add.cxx \
    tblis/internal/0/mult.cxx \
    tblis/internal/0/reduce.cxx \
    \
    tblis/internal/1t/dense/add.cxx \
    tblis/internal/1t/dense/dot.cxx \
    tblis/internal/1t/dense/reduce.cxx \
    tblis/internal/1t/dense/scale.cxx \
    tblis/internal/1t/dense/set.cxx \
    tblis/internal/1t/dense/shift.cxx \
    \
    tblis/internal/1t/dpd/add.cxx \
    tblis/internal/1t/dpd/dot.cxx \
    tblis/internal/1t/dpd/reduce.cxx \
    tblis/internal/1t/dpd/scale.cxx \
    tblis/internal/1t/dpd/set.cxx \
    tblis/internal/1t/dpd/shift.cxx \
    \
    tblis/internal/1t/indexed/add.cxx \
    tblis/internal/1t/indexed/dot.cxx \
    tblis/internal/1t/indexed/reduce.cxx \
    tblis/internal/1t/indexed/scale.cxx \
    tblis/internal/1t/indexed/set.cxx \
    tblis/internal/1t/indexed/shift.cxx \
    \
    tblis/internal/1t/indexed_dpd/add.cxx \
    tblis/internal/1t/indexed_dpd/dot.cxx \
    tblis/internal/1t/indexed_dpd/reduce.cxx \
    tblis/internal/1t/indexed_dpd/scale.cxx \
    tblis/internal/1t/indexed_dpd/set.cxx \
    tblis/internal/1t/indexed_dpd/shift.cxx \
    \
    tblis/internal/3t/dense/mult.cxx \
    \
    tblis/internal/3t/dpd/mult.cxx \
    \
    tblis/internal/3t/indexed/mult.cxx \
    \
    tblis/internal/3t/indexed_dpd/mult.cxx \
    \
    tblis/internal/dense.cxx \
    tblis/internal/dpd.cxx \
    tblis/internal/cpuid.cxx \
    tblis/internal/gemm_thread.cxx \
    tblis/internal/thread.cxx \
    \
    tblis/base/check.cxx \
    tblis/base/types.cxx \
    tblis/base/configs.cxx \
    tblis/base/env.cxx \
    tblis/base/thread.cxx

pkginclude_HEADERS = tblis/tblis.h tblis/config.h

baseincludedir = $(pkgincludedir)/base
baseinclude_HEADERS = \
    \
    tblis/base/types.h \
    tblis/base/check.h \
    tblis/base/configs.h \
    tblis/base/env.h \
    tblis/base/kernels.h \
    tblis/base/macros.h \
    tblis/base/thread.h

iface1tincludedir = $(pkgincludedir)/iface/1t
iface1tinclude_HEADERS = \
    \
    tblis/iface/1t/add.h \
    tblis/iface/1t/dot.h \
    tblis/iface/1t/reduce.h \
    tblis/iface/1t/scale.h \
    tblis/iface/1t/set.h \
    tblis/iface/1t/shift.h

iface3tincludedir = $(pkgincludedir)/iface/3t
iface3tinclude_HEADERS = \
    \
    tblis/iface/3t/mult.h

noinst_LTLIBRARIES =
lib_libtblis_la_LIBADD = external/tci/lib/libtci.la

if IS_CLANG
CXXFLAGS += -Wno-pass-failed
endif

if ENABLE_REFERENCE
noinst_LTLIBRARIES += lib/libreference.la
lib_libtblis_la_LIBADD += lib/libreference.la
lib_libreference_la_SOURCES = tblis/configs/reference/config.cxx
lib_libreference_la_CFLAGS = -O3
lib_libreference_la_CXXFLAGS = -O3
endif

#
# AMD architectures
#

if ENABLE_BULLDOZER
noinst_LTLIBRARIES += lib/libbulldozer.la
lib_libtblis_la_LIBADD += lib/libbulldozer.la
lib_libtblis_la_SOURCES += tblis/configs/bulldozer/config.cxx
lib_libbulldozer_la_SOURCES = tblis/configs/bulldozer/bli_gemm_asm_d4x6_fma4.c \
                             tblis/configs/bulldozer/config_ker.cxx
lib_libbulldozer_la_CFLAGS = -O3 -mavx -mfma4 -march=bdver1 -mfpmath=sse
lib_libbulldozer_la_CXXFLAGS = -O3 -mavx -mfma4 -march=bdver1 -mfpmath=sse
endif

if ENABLE_PILEDRIVER
noinst_LTLIBRARIES += lib/libpiledriver.la
lib_libtblis_la_LIBADD += lib/libpiledriver.la
lib_libtblis_la_SOURCES += tblis/configs/piledriver/config.cxx
lib_libpiledriver_la_SOURCES = tblis/configs/piledriver/bli_gemm_asm_d8x3.c \
                               tblis/configs/piledriver/config_ker.cxx
lib_libpiledriver_la_CFLAGS = -O3 -mavx -mfma -mfma4 -march=bdver2 -mfpmath=sse -mno-tbm
lib_libpiledriver_la_CXXFLAGS = -O3 -mavx -mfma -mfma4 -march=bdver2 -mfpmath=sse -mno-tbm
endif

if ENABLE_EXCAVATOR
noinst_LTLIBRARIES += lib/libexcavator.la
lib_libtblis_la_LIBADD += lib/libexcavator.la
lib_libtblis_la_SOURCES += tblis/configs/excavator/config.cxx
lib_libexcavator_la_SOURCES = tblis/configs/excavator/config_ker.cxx
if !ENABLE_PILEDRIVER
lib_libexcavator_la_SOURCES += tblis/configs/excavator/bli_gemm_asm_d8x3.c
endif
lib_libexcavator_la_CFLAGS = -O3 -mavx -mavx2 -mfma -march=bdver4 -mfpmath=sse -mno-tbm
lib_libexcavator_la_CXXFLAGS = -O3 -mavx -mavx2 -mfma -march=bdver4 -mfpmath=sse -mno-tbm
endif

if ENABLE_ZEN
noinst_LTLIBRARIES += lib/libzen.la
lib_libtblis_la_LIBADD += lib/libzen.la
lib_libtblis_la_SOURCES += tblis/configs/zen/config.cxx
lib_libzen_la_SOURCES = tblis/configs/zen/config_ker.cxx
if !ENABLE_HASWELL
lib_libzen_la_SOURCES += tblis/configs/haswell/bli_gemm_asm_d6x8.c
endif
lib_libzen_la_CFLAGS = -O3 -mavx -mavx2 -mfma -march=znver1 -mfpmath=sse
lib_libzen_la_CXXFLAGS = -O3 -mavx -mavx2 -mfma -march=znver1 -mfpmath=sse
endif

#
# Intel architectures
#

if ENABLE_CORE2
noinst_LTLIBRARIES += lib/libcore2.la
lib_libtblis_la_LIBADD += lib/libcore2.la
lib_libtblis_la_SOURCES += tblis/configs/core2/config.cxx
lib_libcore2_la_SOURCES = tblis/configs/core2/bli_gemm_asm_d4x4.c \
                          tblis/configs/core2/config_ker.cxx
if ENABLE_INTEL_COMPILER
lib_libcore2_la_CFLAGS = -O3 -xSSSE3
lib_libcore2_la_CXXFLAGS = -O3 -xSSSE3
else
lib_libcore2_la_CFLAGS = -O3 -msse3 -mssse3 -march=core2 -mfpmath=sse
lib_libcore2_la_CXXFLAGS = -O3 -msse3 -mssse3 -march=core2 -mfpmath=sse
endif
endif

if ENABLE_SANDYBRIDGE
noinst_LTLIBRARIES += lib/libsandybridge.la
lib_libtblis_la_LIBADD += lib/libsandybridge.la
lib_libtblis_la_SOURCES += tblis/configs/sandybridge/config.cxx
lib_libsandybridge_la_SOURCES = tblis/configs/sandybridge/bli_gemm_asm_d8x4.c \
                                tblis/configs/sandybridge/config_ker.cxx
if ENABLE_INTEL_COMPILER
lib_libsandybridge_la_CFLAGS = -O3 -xAVX
lib_libsandybridge_la_CXXFLAGS = -O3 -xAVX
else
lib_libsandybridge_la_CFLAGS = -O3 -mavx -march=corei7-avx -mfpmath=sse
lib_libsandybridge_la_CXXFLAGS = -O3 -mavx -march=corei7-avx -mfpmath=sse
endif
endif

if ENABLE_HASWELL
noinst_LTLIBRARIES += lib/libhaswell.la
lib_libtblis_la_LIBADD += lib/libhaswell.la
lib_libtblis_la_SOURCES += tblis/configs/haswell/config.cxx
lib_libhaswell_la_SOURCES = tblis/configs/haswell/bli_gemm_asm_d6x8.c \
                            tblis/configs/haswell/config_ker.cxx
#                            tblis/configs/haswell/bli_gemm_asm_d12x4.c \
#                            tblis/configs/haswell/bli_gemm_asm_d8x6.c \
#                            tblis/configs/haswell/bli_gemm_asm_d4x12.c
if ENABLE_INTEL_COMPILER
lib_libhaswell_la_CFLAGS = -O3 -xCORE-AVX2
lib_libhaswell_la_CXXFLAGS = -O3 -xCORE-AVX2
else
lib_libhaswell_la_CFLAGS = -O3 -mavx -mavx2 -mfma -march=core-avx2 -mfpmath=sse
lib_libhaswell_la_CXXFLAGS = -O3 -mavx -mavx2 -mfma -march=core-avx2 -mfpmath=sse
endif
endif

if ENABLE_KNL
noinst_LTLIBRARIES += lib/libknl.la
lib_libtblis_la_LIBADD += lib/libknl.la
lib_libtblis_la_SOURCES += tblis/configs/knl/config.cxx
lib_libknl_la_SOURCES = tblis/configs/knl/bli_spackm_opt_24x16.c \
                        tblis/configs/knl/bli_dpackm_opt_24x8.c \
                        tblis/configs/knl/bli_dpackm_opt_30x8.c \
                        tblis/configs/knl/bli_sgemm_opt_24x16.c \
                        tblis/configs/knl/bli_dgemm_opt_24x8.c \
                        tblis/configs/knl/config_ker.cxx
#                        tblis/configs/knl/bli_dgemm_opt_12x16.c \
#                        tblis/configs/knl/bli_dgemm_opt_30x8.c \
#                        tblis/configs/knl/bli_dgemm_opt_8x24.c \
#                        tblis/configs/knl/bli_sgemm_opt_30x16_knc.c \
#                        tblis/configs/knl/bli_dgemm_opt_30x8_knc.c
if ENABLE_INTEL_COMPILER
lib_libknl_la_CFLAGS = -O3 -xMIC-AVX512
lib_libknl_la_CXXFLAGS = -O3 -xMIC-AVX512
else
if IS_OSX
lib_libknl_la_CFLAGS = -O3 -mavx512f -mavx512pf -march=knl -mfpmath=sse -Wa,-march=knl
lib_libknl_la_CXXFLAGS = -O3 -mavx512f -mavx512pf -march=knl -mfpmath=sse -Wa,-march=knl
else
lib_libknl_la_CFLAGS = -O3 -mavx512f -mavx512pf -march=knl -mfpmath=sse
lib_libknl_la_CXXFLAGS = -O3 -mavx512f -mavx512pf -march=knl -mfpmath=sse
endif
endif
endif

if ENABLE_SKX1
noinst_LTLIBRARIES += lib/libskx1.la
lib_libtblis_la_LIBADD += lib/libskx1.la
lib_libtblis_la_SOURCES += tblis/configs/skx1/config.cxx
lib_libskx1_la_SOURCES = tblis/configs/skx1/config_ker.cxx
if !ENABLE_HASWELL
lib_libskx1_la_SOURCES += tblis/configs/haswell/bli_gemm_asm_d6x8.c
endif
if ENABLE_INTEL_COMPILER
lib_libskx1_la_CFLAGS = -O3 -xCORE-AVX512
lib_libskx1_la_CXXFLAGS = -O3 -xCORE-AVX512
else
if IS_OSX
lib_libskx1_la_CFLAGS = -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -march=skylake-avx512 -mfpmath=sse -Wa,-march=skylake-avx512
lib_libskx1_la_CXXFLAGS = -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -march=skylake-avx512 -mfpmath=sse -Wa,-march=skylake-avx512
else
lib_libskx1_la_CFLAGS = -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -march=skylake-avx512 -mfpmath=sse
lib_libskx1_la_CXXFLAGS = -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -march=skylake-avx512 -mfpmath=sse
endif
endif
endif

if ENABLE_SKX2
noinst_LTLIBRARIES += lib/libskx2.la
lib_libtblis_la_LIBADD += lib/libskx2.la
lib_libtblis_la_SOURCES += tblis/configs/skx2/config.cxx
lib_libskx2_la_SOURCES = tblis/configs/skx2/bli_sgemm_opt_12x32_l2.c \
                         tblis/configs/skx2/bli_dgemm_opt_12x16_l2.c \
                         tblis/configs/skx2/config_ker.cxx
#                        tblis/configs/skx2/bli_dgemm_opt_12x16_l1.c \
#                         tblis/configs/skx2/bli_dgemm_opt_8x8_l1.c \
#                         tblis/configs/skx2/bli_dgemm_opt_8x8_l2.c \
#                         tblis/configs/skx2/bli_dgemm_opt_8x24_l1.c \
#                         tblis/configs/skx2/bli_dgemm_opt_8x24_l2.c \
#                         tblis/configs/skx2/bli_dgemm_opt_6x32_l1.c \
#                         tblis/configs/skx2/bli_dgemm_opt_6x32_l2.c \
#                         tblis/configs/skx2/bli_dgemm_opt_24x8_knl.c
if ENABLE_INTEL_COMPILER
lib_libskx2_la_CFLAGS = -O3 -xCORE-AVX512
lib_libskx2_la_CXXFLAGS = -O3 -xCORE-AVX512
else
if IS_OSX
lib_libskx2_la_CFLAGS = -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -march=skylake-avx512 -mfpmath=sse -Wa,-march=skylake-avx512
lib_libskx2_la_CXXFLAGS = -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -march=skylake-avx512 -mfpmath=sse -Wa,-march=skylake-avx512
else
lib_libskx2_la_CFLAGS = -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -march=skylake-avx512 -mfpmath=sse
lib_libskx2_la_CXXFLAGS = -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -march=skylake-avx512 -mfpmath=sse
endif
endif
endif

#
# ARM architectures
#

if ENABLE_FIRESTORM
noinst_LTLIBRARIES += lib/libfirestorm.la
lib_libtblis_la_LIBADD += lib/libfirestorm.la
lib_libtblis_la_SOURCES += tblis/configs/firestorm/config.cxx
lib_libfirestorm_la_SOURCES = tblis/configs/firestorm/bli_gemm_armv8a_asm_d8x6r.c \
                              tblis/configs/firestorm/config_ker.cxx
lib_libfirestorm_la_CFLAGS = -O3 -march=armv8-a
lib_libfirestorm_la_CXXFLAGS = -O3 -march=armv8-a
endif

noinst_PROGRAMS = bin/test
if ENABLE_BLAS
noinst_PROGRAMS += bin/bench bin/batched_bench #bin/dpd_bench
if ENABLE_SKX1
noinst_PROGRAMS += bin/skx_bench
else
if ENABLE_SKX2
noinst_PROGRAMS += bin/skx_bench
endif
endif
endif

bin_test_SOURCES = test/test.cxx \
                   test/random.cxx \
                   \
                   test/1t/dot.cxx \
                   test/1t/reduce.cxx \
                   test/1t/scale.cxx \
                   test/1t/replicate.cxx \
                   test/1t/trace.cxx \
                   test/1t/transpose.cxx \
                   \
                   test/3m/gemm_ukr.cxx \
                   test/3m/gemm.cxx \
                   test/3m/gemv.cxx \
                   test/3m/ger.cxx \
                   \
                   test/3t/contract.cxx \
                   test/3t/mult.cxx \
                   test/3t/outer_prod.cxx \
                   test/3t/weight.cxx
bin_skx_bench_SOURCES = test/skx_bench.cxx
bin_bench_SOURCES = test/bench.cxx
bin_batched_bench_SOURCES = test/batched_bench.cxx
bin_dpd_bench_SOURCES = test/dpd_bench.cxx

VPATH += $(srcdir)

SUBDIRS = external/tci
ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(srcdir) -I$(srcdir)/tblis -I. -I$(srcdir)/external/tci -Iexternal/tci -I$(srcdir)/external/stl_ext -I$(srcdir)/external/marray -I$(srcdir)/external/catch
AM_LDFLAGS = -pthread
bin_test_LDADD = lib/libtblis.la
bin_skx_bench_LDADD = lib/libtblis.la $(BLAS_LIBS)
bin_bench_LDADD = lib/libtblis.la $(BLAS_LIBS)
bin_batched_bench_LDADD = lib/libtblis.la $(BLAS_LIBS)
bin_dpd_bench_LDADD = lib/libtblis.la $(BLAS_LIBS)

DISTCLEANFILES = tblis/config.h
