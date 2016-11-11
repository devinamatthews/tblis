lib_LTLIBRARIES = lib/libtblis.la
lib_libtblis_la_SOURCES = \
	\
    src/iface/1v/add.cxx \
    src/iface/1v/dot.cxx \
    src/iface/1v/reduce.cxx \
    src/iface/1v/scale.cxx \
    src/iface/1v/set.cxx \
    \
    src/iface/1m/add.cxx \
    src/iface/1m/dot.cxx \
    src/iface/1m/reduce.cxx \
    src/iface/1m/scale.cxx \
    src/iface/1m/set.cxx \
	\
    src/iface/1t/add.cxx \
    src/iface/1t/dot.cxx \
    src/iface/1t/reduce.cxx \
    src/iface/1t/scale.cxx \
    src/iface/1t/set.cxx \
    \
    src/iface/3m/mult.cxx \
    \
    src/iface/3t/mult.cxx \
	\
    src/internal/1v/add.cxx \
    src/internal/1v/dot.cxx \
    src/internal/1v/reduce.cxx \
    src/internal/1v/scale.cxx \
    src/internal/1v/set.cxx \
    \
    src/internal/1m/add.cxx \
    src/internal/1m/dot.cxx \
    src/internal/1m/reduce.cxx \
    src/internal/1m/scale.cxx \
    src/internal/1m/set.cxx \
	\
    src/internal/1t/add.cxx \
    src/internal/1t/dot.cxx \
    src/internal/1t/reduce.cxx \
    src/internal/1t/scale.cxx \
    src/internal/1t/set.cxx \
    \
    src/internal/3m/mult.cxx \
    \
    src/internal/3t/mult.cxx \
    \
    src/configs/configs.cxx \
    \
    src/util/cpuid.cxx \
    src/util/gemm_thread.cxx \
    src/util/random.cxx \
    src/util/thread.cxx
    
pkginclude_HEADERS = src/tblis.h src/tblis_config.h

utilincludedir = $(pkgincludedir)/util
utilinclude_HEADERS = \
	\
	src/util/assert.h \
	src/util/basic_types.h \
	src/util/thread.h

memoryincludedir = $(pkgincludedir)/memory
memoryinclude_HEADERS = \
	\
	src/memory/aligned_allocator.hpp \
	src/memory/stack_allocator.hpp

iface1vincludedir = $(pkgincludedir)/iface/1v
iface1vinclude_HEADERS = \
	\
	src/iface/1v/add.h \
	src/iface/1v/dot.h \
	src/iface/1v/reduce.h \
	src/iface/1v/scale.h \
	src/iface/1v/set.h

iface1mincludedir = $(pkgincludedir)/iface/1m
iface1minclude_HEADERS = \
	\
	src/iface/1m/add.h \
	src/iface/1m/dot.h \
	src/iface/1m/reduce.h \
	src/iface/1m/scale.h \
	src/iface/1m/set.h

iface1tincludedir = $(pkgincludedir)/iface/1t
iface1tinclude_HEADERS = \
	\
	src/iface/1t/add.h \
	src/iface/1t/dot.h \
	src/iface/1t/reduce.h \
	src/iface/1t/scale.h \
	src/iface/1t/set.h

iface3mincludedir = $(pkgincludedir)/iface/3m
iface3minclude_HEADERS = \
	\
	src/iface/3m/mult.h

iface3tincludedir = $(pkgincludedir)/iface/3t
iface3tinclude_HEADERS = \
	\
	src/iface/3t/mult.h
	
marrayincludedir = $(pkgincludedir)/external/marray/include
marrayinclude_HEADERS = \
	\
	src/external/marray/include/marray.hpp \
	src/external/marray/include/miterator.hpp \
	src/external/marray/include/varray.hpp \
	src/external/marray/include/viterator.hpp \
	src/external/marray/include/utility.hpp
	
stl_extincludedir = $(pkgincludedir)/external/stl_ext/include
stl_extinclude_HEADERS = \
	\
	src/external/stl_ext/include/algorithm.hpp \
	src/external/stl_ext/include/any.hpp \
	src/external/stl_ext/include/bounded_vector.hpp \
	src/external/stl_ext/include/complex.hpp \
	src/external/stl_ext/include/cosort.hpp \
	src/external/stl_ext/include/fill_iterator.hpp \
	src/external/stl_ext/include/global_ptr.hpp \
	src/external/stl_ext/include/iostream.hpp \
	src/external/stl_ext/include/ptr_list.hpp \
	src/external/stl_ext/include/ptr_vector.hpp \
	src/external/stl_ext/include/string.hpp \
	src/external/stl_ext/include/type_traits.hpp \
	src/external/stl_ext/include/vector.hpp \
	src/external/stl_ext/include/zip.hpp
	
noinst_LTLIBRARIES =
lib_libtblis_la_LIBADD = src/external/tci/lib/libtci.la

noinst_LTLIBRARIES += lib/libreference.la
lib_libtblis_la_LIBADD += lib/libreference.la
lib_libreference_la_SOURCES = src/configs/reference/config.cxx
lib_libreference_la_CFLAGS = -O3
lib_libreference_la_CXXFLAGS = -O3

#
# AMD architectures
#

if ENABLE_BULLDOZER
noinst_LTLIBRARIES += lib/libbulldozer.la
lib_libtblis_la_LIBADD += lib/libbulldozer.la
lib_libbulldozer_la_SOURCES = src/configs/bulldozer/bli_gemm_asm_d4x6_fma4.c \
					         src/configs/bulldozer/config.cxx
lib_libbulldozer_la_CFLAGS = -O3 -mavx -mfma4 -march=bdver1 -mfpmath=sse
lib_libbulldozer_la_CXXFLAGS = -O3 -mavx -mfma4 -march=bdver1 -mfpmath=sse
endif

if ENABLE_PILEDRIVER
noinst_LTLIBRARIES += lib/libpiledriver.la
lib_libtblis_la_LIBADD += lib/libpiledriver.la
lib_libpiledriver_la_SOURCES = src/configs/piledriver/bli_gemm_asm_d8x3.c \
					           src/configs/piledriver/config.cxx
lib_libpiledriver_la_CFLAGS = -O3 -mavx -mfma -mfma4 -march=bdver2 -mfpmath=sse
lib_libpiledriver_la_CXXFLAGS = -O3 -mavx -mfma -mfma4 -march=bdver2 -mfpmath=sse
endif

if ENABLE_EXCAVATOR
noinst_LTLIBRARIES += lib/libexcavator.la
lib_libtblis_la_LIBADD += lib/libexcavator.la
if !ENABLE_PILEDRIVER
lib_libexcavator_la_SOURCES = src/configs/excavator/bli_gemm_asm_d8x3.c \
					          src/configs/excavator/config.cxx
else
lib_libexcavator_la_SOURCES = src/configs/excavator/config.cxx
endif
lib_libexcavator_la_CFLAGS = -O3 -mavx -mavx2 -mfma -march=bdver4 -mfpmath=sse
lib_libexcavator_la_CXXFLAGS = -O3 -mavx -mavx2 -mfma -march=bdver4 -mfpmath=sse
endif

#
# Intel architectures
#

if ENABLE_CORE2
noinst_LTLIBRARIES += lib/libcore2.la
lib_libtblis_la_LIBADD += lib/libcore2.la
lib_libcore2_la_SOURCES = src/configs/core2/bli_gemm_asm_d4x4.c \
					     src/configs/core2/config.cxx
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
lib_libsandybridge_la_SOURCES = src/configs/sandybridge/bli_gemm_asm_d8x4.c \
					           src/configs/sandybridge/config.cxx
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
lib_libhaswell_la_SOURCES = src/configs/haswell/bli_gemm_asm_d12x4.c \
						   src/configs/haswell/bli_gemm_asm_d8x6.c \
						   src/configs/haswell/bli_gemm_asm_d6x8.c \
						   src/configs/haswell/bli_gemm_asm_d4x12.c \
					       src/configs/haswell/config.cxx
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
lib_libknl_la_SOURCES = src/configs/knl/bli_packm_opt_24x8.c \
					   src/configs/knl/bli_packm_opt_30x8.c \
					   src/configs/knl/bli_dgemm_opt_12x16.c \
					   src/configs/knl/bli_dgemm_opt_24x8.c \
					   src/configs/knl/bli_dgemm_opt_30x8.c \
					   src/configs/knl/bli_dgemm_opt_8x24.c \
					   src/configs/knl/bli_dgemm_opt_30x8_knc.c \
					   src/configs/knl/bli_sgemm_opt_30x16_knc.c \
					   src/configs/knl/config.cxx
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

#if ENABLE_MIC
#noinst_LTLIBRARIES += lib/libmic.la
#lib_libtblis_la_LIBADD += lib/libmic.la
#lib_libmic_la_SOURCES = src/external/blis/config/mic/kernels/3/bli_dgemm_opt_30x8.c \
#					    src/external/blis/config/mic/kernels/3/bli_sgemm_opt_30x16.c
#lib_libmic_la_CFLAGS = -Isrc/external/blis/config/mic -fasm-blocks
#endif

#
# ARM architectures
# 
	
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
# IBM architectures
#

#if ENABLE_POWER7
#noinst_LTLIBRARIES += lib/libpower7.la
#lib_libtblis_la_LIBADD += lib/libpower7.la
#lib_libpower7_la_SOURCES = src/external/blis/config/power7/kernels/3/bli_gemm_opt_8x4.c
#lib_libpower7_la_CFLAGS = -Isrc/external/blis/config/power7 -mcpu=power7 -mtune=power7 -mvsx
#endif
#
#if ENABLE_BGQ
#noinst_LTLIBRARIES += lib/libbgq.la
#lib_libtblis_la_LIBADD += lib/libbgq.la
#lib_libbgq_la_SOURCES = src/external/blis/config/bgq/kernels/3/bli_gemm_int_8x8.c
#lib_libbgq_la_CFLAGS = -Isrc/external/blis/config/bgq -qasm=gcc -qkeyword=asm -qprefetch -qunroll=yes
#endif

#
# MIPS architectures
#

#if ENABLE_LOONGSON3A
#noinst_LTLIBRARIES += lib/libloongson3a.la
#lib_libtblis_la_LIBADD += lib/libloongson3a.la
#lib_libloongson3a_la_SOURCES = src/external/blis/config/loongson3a/kernels/3/bli_gemm_opt_d4x4.c
#lib_libloongson3a_la_CFLAGS = -Isrc/external/blis/config/loongson3a -march=loongson3a -mtune=loongson3a
#endif

noinst_PROGRAMS = bin/test
if ENABLE_BLAS
noinst_PROGRAMS += bin/bench
endif
bin_test_SOURCES = test/test.cxx
bin_bench_SOURCES = test/bench.cxx

VPATH += $(srcdir)

SUBDIRS = src/external/tci
ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(srcdir) -I$(srcdir)/src -I. -Isrc -Isrc/util -Isrc/external/tci/src
AM_LDFLAGS = -pthread
bin_test_LDADD = lib/libtblis.la
bin_bench_LDADD = lib/libtblis.la $(BLAS_LIBS)
