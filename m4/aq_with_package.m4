#
# SYNOPSIS
#
#   AQ_WITH_PACKAGE(PACKAGE, [GIT-REPO],
#         [HEADERS], [INCLUDE-DIRS = "include"],
#         [SYMBOLS], [LIBS], [LIB-DIRS = "lib lib64"],
#         [EXTRA-FLAGS])
#
# LICENSE
#
#   Copyright (c) 2015 Devin Matthews <dmatthews@utexas.edu>
#
#   Copying and distribution of this file, with or without modification, are
#   permitted in any medium without royalty provided the copyright notice
#   and this notice are preserved. This file is offered as-is, without any
#   warranty.
#

AC_DEFUN([AQ_WITH_PACKAGE],
[
    download=no
    downloaded=no
    AC_ARG_WITH(m4_tolower([$1]),
                [AS_HELP_STRING([--with-]m4_tolower([$1])[=<dir>],
                                [Specify the location of $1.])],
                [],
                [download=yes])
    m4_ifval([$6],
    [
        libraries=
        AC_ARG_WITH(m4_tolower([$1])[-libs],
                    [AS_HELP_STRING([--with-]m4_tolower([$1])[-libs=<...>],
                                    [Specify the libraries to link for $1.])],
                    [],
                    [libraries="$6"])
        if test x"$libraries" == x; then
            AS_VAR_COPY([libraries], [with_]m4_tolower(m4_translit([$1], [-+.], [___]))[_libs])
        fi
    ])
    AS_VAR_COPY([pkg_dir], [with_]m4_tolower(m4_translit([$1], [-+.], [___])))
    if test x"$pkg_dir" = xno; then
        m4_ifval([$2], [AC_MSG_ERROR([$1 is a required package.])])
        include_package=no
    elif test x"$download" = xyes && test x"$2" = x; then
        include_package=no
    else
        if test x"$download" = xyes && test x"$2" != x; then
            AS_VAR_SET([pkg_dir], [src/external/]m4_tolower([$1]))
            if ! test -d $pkg_dir; then
                AC_MSG_NOTICE([downloading $1 from external Git repository...])
                if ! git clone -q $2 $pkg_dir; then
                    AC_MSG_ERROR([could not download $1 repository])
                fi
            fi
            if test x"$5$6" != x && test -f $pkg_dir/configure; then
                AC_MSG_NOTICE([configuring $1 in $pkg_dir])
                ( cd $pkg_dir && ./configure )
                AC_MSG_NOTICE([done configuring $1])
            fi
            downloaded=yes
        fi
        m4_ifval([$3],
        [
            include_flags=
            for dir_path in m4_default([$4],[include]); do
                AS_VAR_APPEND([include_flags],[" -I$pkg_dir/$dir_path"])
            done
            for header in $3; do
                AQ_CHECK_HEADER_WITH_PATH([$header],
                                          [],
                                          [AC_MSG_FAILURE([Could not find $header.])],
                                          [$include_flags $8],
                                          [AC_INCLUDES_DEFAULT()])
            done
            m4_tolower(m4_translit([$1], [-+.], [___]))_INCLUDES=$include_flags
            AC_SUBST(m4_tolower(m4_translit([$1], [-+.], [___]))_INCLUDES)
        ])
        if test x"$5" != x; then
            lib_flags=
            for dir_path in m4_default([$7],[lib lib64]); do
                AS_VAR_APPEND([lib_flags],[" -L$pkg_dir/$dir_path"])
            done
            AS_VAR_APPEND([lib_flags], [" $libraries"])
            if test x"$downloaded" != xyes; then
                for symbol in $5; do
                    AQ_CHECK_FUNC_WITH_PATH([$symbol],
                                            [],
                                            [AC_MSG_FAILURE([Could not find $symbol in $libraries.])],
                                            [$lib_flags $8 $LAPACK_LIBS $BLAS_LIBS])
                done
            fi
            m4_tolower(m4_translit([$1], [-+.], [___]))_LIBS=$lib_flags
            AC_SUBST(m4_tolower(m4_translit([$1], [-+.], [___]))_LIBS)
        fi
        include_package=yes
    fi
    AS_IF([test x"$include_package" = xyes], [AC_DEFINE(AS_TR_CPP([HAVE_]$1), [1], [Define if $1 is to be used.])])
    AM_CONDITIONAL(AS_TR_CPP([HAVE_]$1), [test x"$include_package" = xyes])
])
