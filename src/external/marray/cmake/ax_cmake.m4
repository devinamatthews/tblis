# SYNOPSIS
#
#   AX_CMAKE_OPTION(feature, help-string)
#
# DESCRIPTION
#
#   This macro adds an option of the type --{enable,disable}-<feature>
#   using AC_ARG_ENABLE. If either option is specified, this information is passed
#   on to CMake as -DENABLE_<FEATURE>={ON,OFF}, respectively. If no option
#   is passed to configure, then no flag is passed to CMake either, and the default
#   value as specific in CMakeLists.txt will be used.
#
# LICENSE
#
#   Copyright (c) 2025, Devin Matthews <damatthews@smu.edu>
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

AC_DEFUN([_AX_CMAKE_OPTION], [
AC_ARG_ENABLE([$1], [$2], [test "x${$3}" = "xno" &&
    ax_cv_cmake_flags="$ax_cv_cmake_flags -D$4=OFF" ||
    ax_cv_cmake_flags="$ax_cv_cmake_flags -D$4=ON"])
])dnl _AX_CMAKE_OPTION

AC_DEFUN([AX_CMAKE_OPTION], [
_AX_CMAKE_OPTION([$1], [$2], m4_translit([enable_$1], [-+.], [___]), m4_translit([enable_$1], [-+.a-z], [___A-Z]))
])dnl AX_CMAKE_OPTION

# SYNOPSIS
#
#   AX_CMAKE_ENABLE(feature, help-string)
#
# DESCRIPTION
#
#   This macro adds an option of the type --enable-<feature>=<arg>
#   using AC_ARG_ENABLE. If the option is specified, this information is passed
#   on to CMake as -DENABLE_<FEATURE>=<arg>. If no option
#   is passed to configure, then no flag is passed to CMake either, and the default
#   value as specific in CMakeLists.txt will be used.
#
# LICENSE
#
#   Copyright (c) 2025, Devin Matthews <damatthews@smu.edu>
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

AC_DEFUN([_AX_CMAKE_ENABLE], [
AC_ARG_ENABLE([$1], [$2], [ax_cv_cmake_flags="$ax_cv_cmake_flags -D$4=${$3}"])
])dnl _AX_CMAKE_ENABLE

AC_DEFUN([AX_CMAKE_ENABLE], [
_AX_CMAKE_ENABLE([$1], [$2], m4_translit([enable_$1], [-+.], [___]), m4_translit([enable_$1], [-+.a-z], [___A-Z]))
])dnl AX_CMAKE_ENABLE

# SYNOPSIS
#
#   AX_CMAKE_WITH(feature, help-string)
#
# DESCRIPTION
#
#   This macro adds an option of the type --with-<feature>=<arg>
#   using AC_ARG_WITH. If the option is specified, this information is passed
#   on to CMake as -D<FEATURE>=<arg>. If no option
#   is passed to configure, then no flag is passed to CMake either, and the default
#   value as specific in CMakeLists.txt will be used.
#
# LICENSE
#
#   Copyright (c) 2025, Devin Matthews <damatthews@smu.edu>
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

AC_DEFUN([_AX_CMAKE_WITH], [
AC_ARG_WITH([$1], [$2], [ax_cv_cmake_flags="$ax_cv_cmake_flags -D$4=${$3}"])
])dnl _AX_CMAKE_WITH

AC_DEFUN([AX_CMAKE_WITH], [
_AX_CMAKE_WITH([$1], [$2], m4_translit([with_$1], [-+.], [___]), m4_translit([$1], [-+.a-z], [___A-Z]))
])dnl AX_CMAKE_WITH

# SYNOPSIS
#
#   AX_CMAKE([extra-args], [env])
#
# DESCRIPTION
#
#   This macro runs CMake, passing through any user-specific options
#   or enable flags (see AX_CMAKE_OPTION and AX_CMAKE_ENABLE), as well
#   as selected compilers via CC and CXX arguments, and finally any
#   install path options such as --prefix. Additional arguments extra-args
#   may also be passed to CMake, and environment variables env are defined
#   for the invocation.
#
#   The ConfigureWrapper package should be included in CMakeLists.txt
#   to emulate the full range of GNU install paths in case the configure
#   front-end is not used.
#
# LICENSE
#
#   Copyright (c) 2025, Devin Matthews <damatthews@smu.edu>
#   Copyright (c) 2013-2014, Richard Wiedenhöft <richard@wiedenhoeft.xyz>
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

AC_DEFUN([AX_CMAKE], [
test -z $CC || ax_cv_cmake_flags="$ax_cv_cmake_flags -DCMAKE_C_COMPILER=$CC"
test -z $CXX || ax_cv_cmake_flags="$ax_cv_cmake_flags -DCMAKE_CXX_COMPILER=$CXX"
$2 cmake \
	-DCMAKE_INSTALL_PREFIX=$prefix \
	-DINSTALL_PREFIX=$prefix \
	-DINSTALL_EXEC_PREFIX=$exec_prefix \
	-DINSTALL_BINDIR=$bindir \
	-DINSTALL_SBINDIR=$sbindir \
	-DINSTALL_LIBEXECDIR=$libexecdir \
	-DINSTALL_SYSCONFDIR=$sysconfdir \
	-DINSTALL_SHAREDSTATEDIR=$sharedstatedir \
	-DINSTALL_LOCALSTATEDIR=$localstatedir \
	-DINSTALL_RUNSTATEDIR=$runstatedir \
	-DINSTALL_LIBDIR=$libdir \
	-DINSTALL_INCLUDEDIR=$includedir \
	-DINSTALL_DATAROOTDIR=$datarootdir \
	-DINSTALL_DATADIR=$datadir \
	-DINSTALL_INFODIR=$infodir \
	-DINSTALL_LOCALEDIR=$localedir \
	-DINSTALL_MANDIR=$mandir \
	-DINSTALL_DOCDIR=$docdir \
	-DINSTALL_HTMLDIR=$htmldir \
	-DINSTALL_DVIDIR=$dvidir \
	-DINSTALL_PDFDIR=$pdfdir \
	-DINSTALL_PSDIR=$psdir \
	$ax_cv_cmake_flags \
	$1 \
    $srcdir
])dnl AX_CMAKE
