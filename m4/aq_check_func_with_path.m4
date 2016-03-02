#
# SYNOPSIS
#
#   AQ_CHECK_FUNC_WITH_PATH(FUNC,
#         [ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND],
#         [LIBS])
#
# DESCRIPTION
#
#   Same as AC_CHECK_HEADER except that additional path(s) can be
#   added to CFLAGS/CXXFLAGS.
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

AC_DEFUN([AQ_CHECK_FUNC_WITH_PATH], [dnl
  ac_save_LIBS="$LIBS"
  LIBS="$LIBS $4"
  AC_CHECK_FUNC([$1],[$2],[$3])
  LIBS="$ac_save_LIBS"
])
