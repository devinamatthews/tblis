#
# SYNOPSIS
#
#   AQ_CHECK_LIB_WITH_PATH(LIBRARY, FUNCTION,
#              [ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND],
#              [PATH], [OTHER-LIBRARIES])
#
# DESCRIPTION
#
#   Same as AC_CHECK_LIB except that additional path(s) can be
#   added to LDFLAGS.
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

AC_DEFUN([AQ_CHECK_LIB_WITH_PATH], [dnl
  ac_save_LIBS="$LIBS"
  LIBS="$LDFLAGS $5"
  AC_CHECK_LIB([$1],[$2],[$3],[$4],[$6])
  LIBS="$ac_save_LIBS"
])
