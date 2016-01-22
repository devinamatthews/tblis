BLIS = /home/dmatthews/build/blis/gcc/dunnington/install
CXX = g++
#CXXFLAGS = -O2 -mfpmath=sse -fomit-frame-pointer -msse2 -march=native -std=c++0x -Wall -Wno-unused-variable -Wno-sign-compare -fopenmp
CXXFLAGS = -g -O0 -std=c++0x -Wall -Wno-unused-variable -Wno-sign-compare -fopenmp -DDEBUG
LDFLAGS = -g -Llib -L$(BLIS)/lib -fopenmp
AR = ar -cr
INCLUDE = -I. -I./include -I$(BLIS)/include/blis

tensor_SRCS = $(shell find src -name '*.cxx')
tensor_OBJS = $(tensor_SRCS:.cxx=.o)

DEP = $(dir $(1)).deps/$(patsubst %.o,%.Po,$(notdir $(1)))
DEPS = $(foreach obj,$(tensor_OBJS) test/test.o,$(call DEP,$(obj)))
DEPFLAGS = -MT $@ -MD -MP -MF $(call DEP,$@)

tensor: $(tensor_OBJS)
	@mkdir -p lib
	$(AR) lib/libtensor.a $^
	
test: tensor test/test.o
	@mkdir -p bin
	$(CXX) $(LDFLAGS) -o bin/test test/test.o -ltensor -lblis

all: tensor test

clean:
	rm -f $(tensor_OBJS) lib/libtensor.a test/test.o bin/test $(DEPS)

%.o: %.cxx Makefile
	@mkdir -p $(dir $(call DEP,$@))
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDE) -c -o $@ $<
	
-include $(DEPS)
