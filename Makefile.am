lib_LTLIBRARIES = lib/libtblis.la
lib_libtblis_la_SOURCES = \
    \
    src/iface/1t/add.cxx \
    src/iface/1t/dot.cxx \
    src/iface/1t/reduce.cxx \
    src/iface/1t/scale.cxx \
    src/iface/1t/set.cxx \
    src/iface/1t/shift.cxx \
    \
    src/iface/3t/mult.cxx \
    \
    src/internal/0/add.cxx \
    src/internal/0/mult.cxx \
    src/internal/0/reduce.cxx \
    \
    src/internal/1t/dense/add.cxx \
    src/internal/1t/dense/dot.cxx \
    src/internal/1t/dense/reduce.cxx \
    src/internal/1t/dense/scale.cxx \
    src/internal/1t/dense/set.cxx \
    src/internal/1t/dense/shift.cxx \
    \
    src/internal/1t/dpd/add.cxx \
    src/internal/1t/dpd/dot.cxx \
    src/internal/1t/dpd/reduce.cxx \
    src/internal/1t/dpd/scale.cxx \
    src/internal/1t/dpd/set.cxx \
    src/internal/1t/dpd/shift.cxx \
    \
    src/internal/1t/indexed/add.cxx \
    src/internal/1t/indexed/dot.cxx \
    src/internal/1t/indexed/reduce.cxx \
    src/internal/1t/indexed/scale.cxx \
    src/internal/1t/indexed/set.cxx \
    src/internal/1t/indexed/shift.cxx \
    \
    src/internal/1t/indexed_dpd/add.cxx \
    src/internal/1t/indexed_dpd/dot.cxx \
    src/internal/1t/indexed_dpd/reduce.cxx \
    src/internal/1t/indexed_dpd/scale.cxx \
    src/internal/1t/indexed_dpd/set.cxx \
    src/internal/1t/indexed_dpd/shift.cxx \
    \
    src/internal/3t/dense/mult.cxx \
    \
    src/internal/3t/dpd/mult.cxx \
    \
    src/internal/3t/indexed/mult.cxx \
    \
    src/internal/3t/indexed_dpd/mult.cxx \
    \
    src/configs/configs.cxx \
    \
    src/util/basic_types.cxx \
    src/util/configs.cxx \
    src/util/gemm_thread.cxx \
    src/util/cpuid.cxx \
    src/util/env.cxx \
    src/util/random.cxx \
    src/util/thread.cxx
    
pkginclude_HEADERS = src/tblis.h src/tblis_config.h

utilincludedir = $(pkgincludedir)/util
utilinclude_HEADERS = \
    \
    src/util/basic_types.h \
    src/util/configs.h \
    src/util/macros.h \
    src/util/thread.h

memoryincludedir = $(pkgincludedir)/memory
memoryinclude_HEADERS = \
    \
    src/memory/aligned_allocator.hpp \
    src/memory/alignment.hpp \
    src/memory/memory_pool.hpp \
    src/memory/stack_allocator.hpp

iface1tincludedir = $(pkgincludedir)/iface/1t
iface1tinclude_HEADERS = \
    \
    src/iface/1t/add.h \
    src/iface/1t/dot.h \
    src/iface/1t/reduce.h \
    src/iface/1t/scale.h \
    src/iface/1t/set.h \
    src/iface/1t/shift.h

iface3tincludedir = $(pkgincludedir)/iface/3t
iface3tinclude_HEADERS = \
    \
    src/iface/3t/mult.h
    
marrayincludedir = $(pkgincludedir)/external/marray/include
marrayinclude_HEADERS = \
    \
    src/external/marray/include/dpd_marray_base.hpp \
    src/external/marray/include/dpd_marray_view.hpp \
    src/external/marray/include/dpd_marray.hpp \
    src/external/marray/include/dpd_range.hpp \
    src/external/marray/include/dpd_varray_base.hpp \
    src/external/marray/include/dpd_varray_view.hpp \
    src/external/marray/include/dpd_varray.hpp \
    src/external/marray/include/expression.hpp \
    src/external/marray/include/indexed_dpd_varray_base.hpp \
    src/external/marray/include/indexed_dpd_varray_view.hpp \
    src/external/marray/include/indexed_dpd_varray.hpp \
    src/external/marray/include/indexed_varray_base.hpp \
    src/external/marray/include/indexed_varray_view.hpp \
    src/external/marray/include/indexed_varray.hpp \
    src/external/marray/include/marray_base.hpp \
    src/external/marray/include/marray_slice.hpp \
    src/external/marray/include/marray_view.hpp \
    src/external/marray/include/marray.hpp \
    src/external/marray/include/miterator.hpp \
    src/external/marray/include/range.hpp \
    src/external/marray/include/rotate.hpp \
    src/external/marray/include/short_vector.hpp \
    src/external/marray/include/utility.hpp \
    src/external/marray/include/varray_base.hpp \
    src/external/marray/include/varray_view.hpp \
    src/external/marray/include/varray.hpp \
    src/external/marray/include/vector_avx.hpp \
    src/external/marray/include/vector_avx512.hpp \
    src/external/marray/include/vector_sse41.hpp \
    src/external/marray/include/vector.hpp \
    src/external/marray/include/viterator.hpp
    
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

if IS_CLANG
CXXFLAGS += -Wno-pass-failed
endif

if ENABLE_REFERENCE
noinst_LTLIBRARIES += lib/libreference.la
lib_libtblis_la_LIBADD += lib/libreference.la
lib_libreference_la_SOURCES = src/configs/reference/config.cxx
lib_libreference_la_CFLAGS = -O3
lib_libreference_la_CXXFLAGS = -O3
endif

#
# AMD architectures
#

if ENABLE_BULLDOZER
noinst_LTLIBRARIES += lib/libbulldozer.la
lib_libtblis_la_LIBADD += lib/libbulldozer.la
lib_libtblis_la_SOURCES += src/configs/bulldozer/config.cxx
lib_libbulldozer_la_SOURCES = src/configs/bulldozer/bli_gemm_asm_d4x6_fma4.c \
                             src/configs/bulldozer/config_ker.cxx
lib_libbulldozer_la_CFLAGS = -O3 -mavx -mfma4 -march=bdver1 -mfpmath=sse
lib_libbulldozer_la_CXXFLAGS = -O3 -mavx -mfma4 -march=bdver1 -mfpmath=sse
endif

if ENABLE_PILEDRIVER
noinst_LTLIBRARIES += lib/libpiledriver.la
lib_libtblis_la_LIBADD += lib/libpiledriver.la
lib_libtblis_la_SOURCES += src/configs/piledriver/config.cxx
lib_libpiledriver_la_SOURCES = src/configs/piledriver/bli_gemm_asm_d8x3.c \
                               src/configs/piledriver/config_ker.cxx
lib_libpiledriver_la_CFLAGS = -O3 -mavx -mfma -mfma4 -march=bdver2 -mfpmath=sse -mno-tbm
lib_libpiledriver_la_CXXFLAGS = -O3 -mavx -mfma -mfma4 -march=bdver2 -mfpmath=sse -mno-tbm
endif

if ENABLE_EXCAVATOR
noinst_LTLIBRARIES += lib/libexcavator.la
lib_libtblis_la_LIBADD += lib/libexcavator.la
lib_libtblis_la_SOURCES += src/configs/excavator/config.cxx
lib_libexcavator_la_SOURCES = src/configs/excavator/config_ker.cxx
if !ENABLE_PILEDRIVER
lib_libexcavator_la_SOURCES += src/configs/excavator/bli_gemm_asm_d8x3.c
endif
lib_libexcavator_la_CFLAGS = -O3 -mavx -mavx2 -mfma -march=bdver4 -mfpmath=sse -mno-tbm
lib_libexcavator_la_CXXFLAGS = -O3 -mavx -mavx2 -mfma -march=bdver4 -mfpmath=sse -mno-tbm
endif

if ENABLE_ZEN
noinst_LTLIBRARIES += lib/libzen.la
lib_libtblis_la_LIBADD += lib/libzen.la
lib_libtblis_la_SOURCES += src/configs/zen/config.cxx
lib_libzen_la_SOURCES = src/configs/zen/config_ker.cxx
if !ENABLE_HASWELL
lib_libzen_la_SOURCES += src/configs/haswell/bli_gemm_asm_d6x8.c
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
lib_libtblis_la_SOURCES += src/configs/core2/config.cxx
lib_libcore2_la_SOURCES = src/configs/core2/bli_gemm_asm_d4x4.c \
                          src/configs/core2/config_ker.cxx
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
lib_libtblis_la_SOURCES += src/configs/sandybridge/config.cxx
lib_libsandybridge_la_SOURCES = src/configs/sandybridge/bli_gemm_asm_d8x4.c \
                                src/configs/sandybridge/config_ker.cxx
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
lib_libtblis_la_SOURCES += src/configs/haswell/config.cxx
lib_libhaswell_la_SOURCES = src/configs/haswell/bli_gemm_asm_d6x8.c \
                            src/configs/haswell/config_ker.cxx
#                            src/configs/haswell/bli_gemm_asm_d12x4.c \
#                            src/configs/haswell/bli_gemm_asm_d8x6.c \
#                            src/configs/haswell/bli_gemm_asm_d4x12.c
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
lib_libtblis_la_SOURCES += src/configs/knl/config.cxx
lib_libknl_la_SOURCES = src/configs/knl/bli_spackm_opt_24x16.c \
                        src/configs/knl/bli_dpackm_opt_24x8.c \
                        src/configs/knl/bli_dpackm_opt_30x8.c \
                        src/configs/knl/bli_sgemm_opt_24x16.c \
                        src/configs/knl/bli_dgemm_opt_24x8.c \
                        src/configs/knl/config_ker.cxx
#                        src/configs/knl/bli_dgemm_opt_12x16.c \
#                        src/configs/knl/bli_dgemm_opt_30x8.c \
#                        src/configs/knl/bli_dgemm_opt_8x24.c \
#                        src/configs/knl/bli_sgemm_opt_30x16_knc.c \
#                        src/configs/knl/bli_dgemm_opt_30x8_knc.c
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
lib_libtblis_la_SOURCES += src/configs/skx1/config.cxx
lib_libskx1_la_SOURCES = src/configs/skx1/config_ker.cxx
if !ENABLE_SKX2
lib_libtblis_la_SOURCES += src/configs/skx2/vpu_count.cxx
endif
if !ENABLE_HASWELL
lib_libskx1_la_SOURCES += src/configs/haswell/bli_gemm_asm_d6x8.c
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
lib_libtblis_la_SOURCES += src/configs/skx2/vpu_count.cxx \
                           src/configs/skx2/config.cxx
lib_libskx2_la_SOURCES = src/configs/skx2/bli_sgemm_opt_12x32_l2.c \
                         src/configs/skx2/bli_dgemm_opt_12x16_l2.c \
                         src/configs/skx2/config_ker.cxx
#                        src/configs/skx2/bli_dgemm_opt_12x16_l1.c \
#                         src/configs/skx2/bli_dgemm_opt_8x8_l1.c \
#                         src/configs/skx2/bli_dgemm_opt_8x8_l2.c \
#                         src/configs/skx2/bli_dgemm_opt_8x24_l1.c \
#                         src/configs/skx2/bli_dgemm_opt_8x24_l2.c \
#                         src/configs/skx2/bli_dgemm_opt_6x32_l1.c \
#                         src/configs/skx2/bli_dgemm_opt_6x32_l2.c \
#                         src/configs/skx2/bli_dgemm_opt_24x8_knl.c
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

#if ENABLE_MIC
#noinst_LTLIBRARIES += lib/libmic.la
#lib_libtblis_la_LIBADD += lib/libmic.la
#lib_libmic_la_SOURCES = src/external/blis/config/mic/kernels/3/bli_dgemm_opt_30x8.c \
#                         src/external/blis/config/mic/kernels/3/bli_sgemm_opt_30x16.c
#lib_libmic_la_CFLAGS = -Isrc/external/blis/config/mic -fasm-blocks
#endif

#
# ARM architectures
# 
    
if ENABLE_ARMV7A
noinst_LTLIBRARIES += lib/libarmv7a.la
lib_libtblis_la_LIBADD += lib/libarmv7a.la
lib_libtblis_la_SOURCES += src/configs/armv7a/config.cxx
lib_libarmv7a_la_SOURCES = src/configs/armv7a/config_ker.cxx \
                           src/configs/armv7a/bli_gemm_armv7a_int_d4x4.c
lib_libarmv7a_la_CCASFLAGS = -mfloat-abi=hard -mfpu=vfpv3 -marm -march=armv7-a
lib_libarmv7a_la_CFLAGS = -mfloat-abi=hard -mfpu=vfpv3 -marm -march=armv7-a
endif

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

if ENABLE_ARMV8A
noinst_LTLIBRARIES += lib/libarmv8a.la
lib_libtblis_la_LIBADD += lib/libarmv8a.la
lib_libtblis_la_SOURCES += src/configs/armv8a/config.cxx
lib_libarmv8a_la_SOURCES = src/configs/armv8a/config_ker.cxx \
                           src/configs/armv8a/bli_gemm_asm_d6x8.c
lib_libarmv8a_la_CFLAGS = -march=armv8-a+fp+simd
endif

if ENABLE_THUNDERX2
noinst_LTLIBRARIES += lib/libthunderx2.la
lib_libtblis_la_LIBADD += lib/libthunderx2.la
lib_libtblis_la_SOURCES += src/configs/thunderx2/config.cxx
lib_libthunderx2_la_SOURCES = src/configs/thunderx2/config_ker.cxx
if !ENABLE_ARMV8A
lib_libthunderx2_la_SOURCES += src/configs/armv8a/bli_gemm_asm_d6x8.c
endif
lib_libthunderx2_la_CFLAGS = -march=armv8-a+fp+simd -mcpu=thunderx2t99
endif

if ENABLE_ARMV8A_SVE512
noinst_LTLIBRARIES += lib/libarmv8a_sve512.la
lib_libtblis_la_LIBADD += lib/libarmv8a_sve512.la
lib_libtblis_la_SOURCES += src/configs/armv8a_sve512/config.cxx
lib_libarmv8a_sve512_la_SOURCES = src/configs/armv8a_sve512/config_ker.cxx \
                                  src/configs/armv8a_sve512/bli_gemm_armsve_asm_2vx10_unindexed.c \
                                  src/configs/armv8a_sve512/bli_dpackm_armsve512_asm_16xk.c \
                                  src/configs/armv8a_sve512/bli_dpackm_armsve512_asm_10xk.c
lib_libarmv8a_sve512_la_CFLAGS = -march=armv8-a+sve
CXXFLAGS += -march=armv8-a+sve
endif

if ENABLE_ARMV8A_SVE256
noinst_LTLIBRARIES += lib/libarmv8a_sve256.la
lib_libtblis_la_LIBADD += lib/libarmv8a_sve256.la
lib_libtblis_la_SOURCES += src/configs/armv8a_sve256/config.cxx
lib_libarmv8a_sve256_la_SOURCES = src/configs/armv8a_sve256/config_ker.cxx \
                                  src/configs/armv8a_sve512/bli_gemm_armsve_asm_2vx10_unindexed.c
lib_libarmv8a_sve256_la_CFLAGS = -march=armv8-a+sve
CXXFLAGS += -march=armv8-a+sve
endif

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

SUBDIRS = src/external/tci
ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(srcdir) -I$(srcdir)/src -I. -Isrc -Isrc/util -I$(srcdir)/src/external/tci -Isrc/external/tci/tci
AM_LDFLAGS = -pthread
bin_test_LDADD = lib/libtblis.la
bin_skx_bench_LDADD = lib/libtblis.la $(BLAS_LIBS)
bin_bench_LDADD = lib/libtblis.la $(BLAS_LIBS)
bin_batched_bench_LDADD = lib/libtblis.la $(BLAS_LIBS)
bin_dpd_bench_LDADD = lib/libtblis.la $(BLAS_LIBS)

DISTCLEANFILES = src/tblis_config.h
