cmake_minimum_required(VERSION 3.23)

file(STRINGS marray-version MARRAY_VERSION_RAW)
set(MARRAY_VERSION "${MARRAY_VERSION_RAW}")
string(REPLACE "." ";" MARRAY_VERSION_RAW ${MARRAY_VERSION_RAW})
list(GET MARRAY_VERSION_RAW 0 MARRAY_VERSION_MAJOR)
list(GET MARRAY_VERSION_RAW 1 MARRAY_VERSION_MINOR)
set(prefix ${CMAKE_INSTALL_PREFIX})

set(CMAKE_C_COMPILER_NAMES clang gcc icc cc)
set(CMAKE_CXX_COMPILER_NAMES clang++ g++ icpc c++ cxx)

project(marray
    VERSION "${MARRAY_VERSION}"
    LANGUAGES CXX
    HOMEPAGE_URL "http://www.github.com/MatthewsResearchGroup/marray"
    DESCRIPTION "Expressive, easy-to-use tensor (multi-dimensional array) interfaces."
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FetchContent)
include(ConfigureWrapper)
include(ExternalProject)
include(CheckCSourceCompiles)
include(CMakePackageConfigHelpers)

option(ENABLE_TESTS "Enable building unit tests." OFF)
option(ENABLE_COVERAGE "Enable code coverage reporting for tests (requires clang)." OFF)

set(MARRAY_HEADERS
    marray/detail/config.hpp
    marray/detail/utility.hpp
    marray/detail/vector_avx.hpp
    marray/detail/vector_avx512.hpp
    marray/detail/vector_neon.hpp
    marray/detail/vector_sse41.hpp
    marray/detail/vector.hpp
    marray/dpd/dpd_marray_base.hpp
    marray/dpd/dpd_marray_view.hpp
    marray/dpd/dpd_marray.hpp
    marray/dpd/dpd_range.hpp
    marray/fwd/expression_fwd.hpp
    marray/fwd/marray_fwd.hpp
    marray/indexed/indexed_marray_base.hpp
    marray/indexed/indexed_marray_view.hpp
    marray/indexed/indexed_marray.hpp
    marray/indexed_dpd/indexed_dpd_marray_base.hpp
    marray/indexed_dpd/indexed_dpd_marray_view.hpp
    marray/indexed_dpd/indexed_dpd_marray.hpp
    marray/array_1d.hpp
    marray/array_2d.hpp
    marray/expression.hpp
    marray/index_iterator.hpp
    marray/marray_base.hpp
    marray/marray_iterator.hpp
    marray/marray_slice.hpp
    marray/marray_view.hpp
    marray/marray.hpp
    marray/range.hpp
    marray/rotate.hpp
    marray/short_vector.hpp
    marray/types.hpp
)

if(ENABLE_TESTS)
    set(MARRAY_SOURCES
        test/dpd_marray_view.cxx
        test/dpd_marray.cxx
        test/dpd_range.cxx
        test/expression.cxx
        test/index_iterator.cxx
        test/indexed_dpd_marray.cxx
        test/indexed_marray.cxx
        test/marray_view.cxx
        test/marray.cxx
        test/range.cxx
        test/short_vector.cxx
    )

    set(MARRAY_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG 2b60af89e23d28eefc081bc930831ee9d45ea58b # v3.8.1
    )
    FetchContent_MakeAvailable(Catch2)

    get_target_property(CATCH2_INCLUDES Catch2::Catch2 INTERFACE_INCLUDE_DIRECTORIES)
    # For some reason there is an empty entry at the end of the list, and all
    # other attempts to get rid of it do not work. If this empty entry ever goes
    # away then this will be wrong.
    list(POP_BACK CATCH2_INCLUDES)
    set(CATCH2_CFLAGS "$<LIST:TRANSFORM,${CATCH2_INCLUDES},PREPEND,-I>")

    if(ENABLE_COVERAGE)
        if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            message(FATAL_ERROR "Clang is required for code coverage analysis.")
        endif()

        get_filename_component(LLVM_DIR ${CMAKE_CXX_COMPILER} PROGRAM)
        get_filename_component(LLVM_DIR ${LLVM_DIR} DIRECTORY)
        set(LLVM_CONFIG "${LLVM_DIR}/llvm-config")

        execute_process(COMMAND
            ${LLVM_CONFIG} "--cxxflags"
            OUTPUT_VARIABLE LLVM_CXXFLAGS
        )
        string(REGEX REPLACE "^[ \t\r\n]+" "" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")
        string(REGEX REPLACE "[ \t\r\n]+$" "" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")
        string(REGEX REPLACE "[ \t\r\n]+" ";" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")

        execute_process(COMMAND
            ${LLVM_CONFIG} "--ldflags" "--libs" "--system-libs"
            OUTPUT_VARIABLE LLVM_LIBS
        )
        string(REGEX REPLACE "^[ \t\r\n]+" "" LLVM_LIBS "${LLVM_LIBS}")
        string(REGEX REPLACE "[ \t\r\n]+$" "" LLVM_LIBS "${LLVM_LIBS}")
        string(REGEX REPLACE "[ \t\r\n]+" ";" LLVM_LIBS "${LLVM_LIBS}")

        execute_process(COMMAND
            ${CMAKE_CXX_COMPILER} "-print-resource-dir"
            OUTPUT_VARIABLE RESOURCE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/test.cxx" "#include <array>")
        execute_process(COMMAND
            ${CMAKE_CXX_COMPILER} -E -v "${CMAKE_CURRENT_BINARY_DIR}/test.cxx"
            OUTPUT_QUIET ERROR_VARIABLE INCLUDE_LIST
        )
        string(REPLACE " (framework directory)" "" INCLUDE_LIST "${INCLUDE_LIST}")
        string(REGEX REPLACE ".*#include <\.\.\.> search starts here:[ \r\n\t]*" "" INCLUDE_LIST "${INCLUDE_LIST}")
        string(REGEX REPLACE "[ \r\n\t]*End of search list.*" "" INCLUDE_LIST "${INCLUDE_LIST}")
        string(REGEX REPLACE "[ \t\r\n]+" ";" INCLUDE_LIST "${INCLUDE_LIST}")
        set(INCLUDE_LIST "$<LIST:TRANSFORM,${INCLUDE_LIST},PREPEND,-isystem;>")

        # Note order is important here
        set(CLANG_LIBS
            -lclangTooling
            -lclangASTMatchers
            -lclangFormat
            -lclangFrontend
            -lclangDriver
            -lclangParse
    	    -lclangSerialization
            -lclangSema
            -lclangEdit
            -lclangAnalysis
            -lclangToolingCore
            -lclangAST
            -lclangRewrite
    	    -lclangLex
            -lclangFrontendTool
            -lclangRewriteFrontend
            -lclangStaticAnalyzerFrontend
    	    -lclangStaticAnalyzerCheckers
            -lclangStaticAnalyzerCore
            -lclangAPINotes
            -lclangBasic
        )

        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15.0.0)
            list(APPEND CLANG_LIBS -lclangSupport)
        endif()

        # Not sure when this is actually needed, but it
        # *is needed* on an MBP (with homebrew clang) but
        # *is not needed* on Linux (e.g. CircleCI)
        if(APPLE)
            list(APPEND CLANG_LIBS -lc++)
        endif()

        if (POLICY CMP0135)
            cmake_policy(SET CMP0135 NEW)
        endif()

        ExternalProject_Add(force-cover
            URL "https://github.com/emilydolson/force-cover/archive/refs/tags/v3.0.tar.gz"
            PREFIX ${CMAKE_CURRENT_BINARY_DIR}
            CONFIGURE_COMMAND ""
            BUILD_IN_SOURCE ON
            BUILD_COMMAND
                ${CMAKE_CXX_COMPILER} -O0 -g -std=c++20
                "${LLVM_CXXFLAGS}"
                "${CMAKE_CURRENT_BINARY_DIR}/src/force-cover/force_cover.cpp"
                -o "${CMAKE_CURRENT_BINARY_DIR}/src/force-cover/force-cover"
                "${CLANG_LIBS}"
                "${LLVM_LIBS}"
            INSTALL_COMMAND ""
            LOG_DOWNLOAD ON
            LOG_BUILD ON
        )
        ExternalProject_Add_StepTargets(force-cover build)

        ExternalProject_Get_property(force-cover SOURCE_DIR)
        set(FORCE_COVER "${SOURCE_DIR}/force-cover")
        set(PROCESSED_DIR "${CMAKE_CURRENT_BINARY_DIR}/src")

        foreach(file IN LISTS MARRAY_HEADERS)
            list(APPEND PROCESSED_HEADERS ${PROCESSED_DIR}/${file})
            add_custom_command(
                OUTPUT "${PROCESSED_DIR}/${file}"
                COMMAND
                    mkdir -p "$$(dirname" "${PROCESSED_DIR}/${file})" &&
                    ${FORCE_COVER} "${CMAKE_CURRENT_SOURCE_DIR}/${file}"
                    -- -xc++ -std=c++20
                    -resource-dir ${RESOURCE_DIR}
                    "${INCLUDE_LIST}"
                    "-I${CMAKE_CURRENT_SOURCE_DIR}"
                    "${CATCH2_CFLAGS}"
                    > "${PROCESSED_DIR}/${file}"
                COMMAND_EXPAND_LISTS
                DEPENDS ${file} force-cover-build
            )
        endforeach()

        foreach(file IN LISTS MARRAY_SOURCES)
            list(APPEND PROCESSED_SOURCES ${PROCESSED_DIR}/${file})
            add_custom_command(
                OUTPUT "${PROCESSED_DIR}/${file}"
                COMMAND
                    mkdir -p "$$(dirname" "${PROCESSED_DIR}/${file})" &&
                    ${FORCE_COVER} "${CMAKE_CURRENT_SOURCE_DIR}/${file}"
                    -- -xc++ -std=c++20
                    -resource-dir ${RESOURCE_DIR}
                    "${INCLUDE_LIST}"
                    "-I${CMAKE_CURRENT_SOURCE_DIR}"
                    "${CATCH2_CFLAGS}"
                    > "${PROCESSED_DIR}/${file}"
                COMMAND_EXPAND_LISTS
                DEPENDS ${file} force-cover-build ${PROCESSED_HEADERS}
            )
        endforeach()

        set(MARRAY_SOURCES ${PROCESSED_SOURCES})
        set(MARRAY_DIR "${PROCESSED_DIR}")
    endif()

    add_executable(marray-test ${MARRAY_SOURCES})
    set_target_properties(marray-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
        RUNTIME_OUTPUT_NAME test
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    target_link_libraries(marray-test Catch2::Catch2WithMain)
    target_include_directories(marray-test PRIVATE ${MARRAY_DIR})

    if(ENABLE_COVERAGE)
        target_compile_options(marray-test PRIVATE -fprofile-instr-generate -fcoverage-mapping)
        target_link_options(marray-test PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    endif()
endif()

add_library(marray INTERFACE)
add_library(MArray::marray ALIAS marray)
target_include_directories(marray INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>
)
target_sources(marray
    INTERFACE FILE_SET headers TYPE HEADERS
    FILES ${MARRAY_HEADERS}
)
install(TARGETS marray EXPORT marray-targets
    FILE_SET headers DESTINATION ${INSTALL_INCLUDEDIR}
)
install(EXPORT marray-targets
    FILE MArrayTargets.cmake
    NAMESPACE MArray::
    DESTINATION ${INSTALL_LIBDIR}/cmake/MArray
)

write_basic_package_version_file(
    "MArrayConfigVersion.cmake"
    VERSION ${CMAKE_PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES "MArrayConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/MArrayConfigVersion.cmake"
    DESTINATION ${INSTALL_LIBDIR}/cmake/MArray
)

export(EXPORT marray-targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/MArrayTargets.cmake
    NAMESPACE MArray::
)

find_program(SPHINX sphinx-build)
find_program(DOXYGEN doxygen)

if (SPHINX AND DOXYGEN)
    add_custom_target(marray-docs
        COMMAND ${SPHINX} "${CMAKE_CURRENT_SOURCE_DIR}/docs" "${CMAKE_CURRENT_BINARY_DIR}docs/sphinx"
    )
else()
    add_custom_target(marray-docs
        COMMAND ""
        COMMENT "sphinx and doxygen (and the sphinx plugins breathe, \
myst_parser, and sphinx-rtd-theme) are required for \
building documentation"
    )
endif()
