AC_INIT(MArray,
    m4_normalize(m4_include([marray-version])),
    damatthews@smu.edu,
    marray,
    http://www.github.com/MatthewsResearchGroup/marray
)

AC_CONFIG_MACRO_DIR([cmake])

AX_CMAKE_OPTION([tests], AS_HELP_STRING([--enable-tests], [Enable unit testing]))

AX_CMAKE_OPTION([coverage], AS_HELP_STRING([--enable-coverage], [Enable code coverage testing with llvm-cov]))

AX_CMAKE()