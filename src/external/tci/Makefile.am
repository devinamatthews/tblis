lib_LTLIBRARIES = lib/libtci.la
lib_libtci_la_SOURCES = src/barrier.c \
						src/communicator.c \
						src/context.c \
						src/mutex.c \
						src/parallel.c
						
include_HEADERS = src/tci.h

VPATH += $(srcdir)

AM_CPPFLAGS = -I$(srcdir) -I$(srcdir)/src -I. -Isrc
ACLOCAL_AMFLAGS = -I m4
