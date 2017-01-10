lib_LTLIBRARIES = lib/libtci.la
lib_libtci_la_SOURCES = tci/barrier.c \
						tci/communicator.c \
						tci/context.c \
						tci/mutex.c \
						tci/parallel.c \
						tci/work_item.c
						
nobase_include_HEADERS = tci.h \
						 tci/tci_config.h \
                         tci/barrier.h \
                         tci/communicator.h \
                         tci/context.h \
                         tci/mutex.h \
                         tci/parallel.h \
                         tci/work_item.h \
                         tci/yield.h

VPATH += $(srcdir)

AM_CPPFLAGS = -I$(srcdir) -I$(srcdir)/tci -I. -Itci
ACLOCAL_AMFLAGS = -I m4
