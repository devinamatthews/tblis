lib_LTLIBRARIES = lib/libtci.la
lib_libtci_la_SOURCES = tci/barrier.c \
						tci/context.c \
						tci/mutex.c \
						tci/parallel.c \
						tci/slot.c \
						tci/work_item.c
						
if CPLUSPLUS
lib_libtci_la_SOURCES += tci/communicator_cpp.cxx \
						 tci/task_set_cpp.cxx
else
lib_libtci_la_SOURCES += tci/communicator.c \
						 tci/task_set.c
endif

nobase_include_HEADERS = tci.h \
						 tci/tci_config.h \
                         tci/barrier.h \
                         tci/communicator.h \
                         tci/context.h \
                         tci/mutex.h \
                         tci/parallel.h \
                         tci/slot.h \
                         tci/work_item.h \
                         tci/yield.h

%_cpp.cxx: %.c
	@cp $< $@

VPATH += $(srcdir)

AM_CPPFLAGS = -I$(srcdir) -I$(srcdir)/tci -I. -Itci
ACLOCAL_AMFLAGS = -I m4
