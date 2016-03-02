#
# SYNOPSIS
#
#   AQ_CHECK_HEADER_WITH_PATH(HEADER-FILE,
#         [ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND],
#         [PATH], [INCLUDES])
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

AC_DEFUN([AQ_CHECK_HEADER_WITH_PATH], [dnl
  ac_save_CPPFLAGS="$CPPFLAGS"
  CPPFLAGS="$CPPFLAGS $4"
  AC_CHECK_HEADER([$1],[$2],[$3],[$5])
  CPPFLAGS="$ac_save_CPPFLAGS"
])
