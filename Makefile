CXX = g++
CXXFLAGS = -g -O0 -std=c++0x -Wall -Wno-sign-compare
LDFLAGS = -g -Llib -L/home/dmatthews/build/blis-reference-debug/install/lib
AR = ar -cr
INCLUDE = -I. -I./include -I/home/dmatthews/build/blis-reference-debug/install/include/blis

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
