VPATH += $(srcdir)

ACLOCAL_AMFLAGS = -I m4

AM_CPPFLAGS = -I$(srcdir)/include -Iexternal/googletest/include

check_PROGRAMS = bin/test
bin_test_SOURCES = test/miterator.cxx test/marray.cxx test/utility.cxx
bin_test_LDADD = lib/libgtest_main.a lib/libgtest.a

TESTS = bin/test

.PHONY: test clean-gtest distclean-gtest

test: check-am

check-am: lib/libgtest_main.a lib/libgtest.a

external/googletest/VERSION:
	wget https://github.com/google/googletest/archive/release-1.7.0.tar.gz -O external/googletest-1.7.0.tar.gz
	tar zxvf external/googletest-1.7.0.tar.gz -C external
	mv external/googletest-release-1.7.0 external/googletest
	echo "1.7.0" > external/googletest/VERSION

lib/libgtest.a: external/googletest/VERSION
	$(CXX) $(CXXFLAGS) -Iexternal/googletest/include -Iexternal/googletest -pthread \
		-o external/googletest/gtest-all.o -c external/googletest/src/gtest-all.cc
	$(AR) rv lib/libgtest.a external/googletest/gtest-all.o

lib/libgtest_main.a: external/googletest/VERSION
	$(CXX) $(CXXFLAGS) -Iexternal/googletest/include -Iexternal/googletest -pthread \
		-o external/googletest/gtest_main.o -c external/googletest/src/gtest_main.cc
	$(AR) rv lib/libgtest_main.a external/googletest/gtest_main.o
	
clean-local: clean-gtest
	
distclean-local: distclean-gtest

clean-gtest:
	rm -f external/googletest/src/*.o
	rm -f lib/libgtest.a
	rm -f lib/libgtest_main.a
	
distclean-gtest:
	rm -f external/googletest-1.7.0.tar.gz
	rm -rf external/googletest
	