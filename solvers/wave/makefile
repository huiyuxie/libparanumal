#####################################################################################
#
# The MIT License (MIT)
#
# Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#####################################################################################

define WAVE_HELP_MSG

Wave solver makefile targets:

   make waveMain (default)
   make lib
   make clean
   make clean-libs
   make clean-kernels
   make realclean
   make info
   make help
   make test

Usage:

make waveMain
   Build waveMain executable.
make lib
   Build libwave.a solver library.
make clean
   Clean the waveMain executable, library, and object files.
make clean-libs
   In addition to "make clean", also clean needed libraries.
make clean-kernels
   In addition to "make clean-libs", also cleans the cached OCCA kernels.
make realclean
   In addition to "make clean-kernels", also clean 3rd party libraries.
make info
   List directories and compiler flags in use.
make help
   Display this help message.
make test
   Run tests.

Can use "make verbose=true" for verbose output.

endef

ifeq (,$(filter waveMain lib clean clean-libs clean-kernels \
                realclean info help test, $(MAKECMDGOALS)))
ifneq (,$(MAKECMDGOALS))
$(error ${WAVE_HELP_MSG})
endif
endif

ifndef LIBP_MAKETOP_LOADED
ifeq (,$(wildcard ../../make.top))
$(error cannot locate ${PWD}/../../make.top)
else
include ../../make.top
endif
endif

# CUDA helpers
NVCC        ?= nvcc
CUDA_HOME   ?= /usr/local/cuda
CUDA_ARCH   ?= $(shell uname -m)

LIBP_INCLUDES += -I$(CUDA_HOME)/include \
                 -I$(CUDA_HOME)/targets/$(CUDA_ARCH)-linux/include

# libraries
ELLIPTIC_DIR    = ${LIBP_DIR}/solvers/elliptic
WAVE_LIBP_LIBS  = parAlmond linearSolver mesh parAdogs ogs linAlg core timeStepper

# includes
INCLUDES  = -I${ELLIPTIC_DIR} ${LIBP_INCLUDES} -I.

# defines
DEFINES   = ${LIBP_DEFINES} \
            -DLIBP_DIR='"${LIBP_DIR}"'

# flags
WAVE_CXXFLAGS = ${LIBP_CXXFLAGS} ${DEFINES} ${INCLUDES}
CUFLAGS       ?= -O3 -std=c++17 -Xcompiler -fPIC ${DEFINES} ${INCLUDES}

LIBS = -L${ELLIPTIC_DIR} -lelliptic \
       -L${LIBP_LIBS_DIR} $(addprefix -l,$(WAVE_LIBP_LIBS)) \
       ${LIBP_LIBS}

CU_SRC := $(wildcard src/*.cu)
ifneq ($(strip $(CU_SRC)),)
  LIBS += -L$(CUDA_HOME)/lib64 -lcudart
endif

LFLAGS = ${WAVE_CXXFLAGS} ${LIBS}

# objects
DEPS = $(wildcard *.hpp) \
       $(wildcard $(LIBP_INCLUDE_DIR)/*.h) \
       $(wildcard $(LIBP_INCLUDE_DIR)/*.hpp) \
       $(wildcard $(ELLIPTIC_DIR)/*.hpp)

SRC     = $(wildcard src/*.cpp)
OBJS    = $(SRC:.cpp=.o)
CU_OBJS = $(CU_SRC:.cu=.o)

.PHONY: all libp_libs libelliptic clean clean-libs \
        clean-kernels realclean help info test

all: waveMain

lib: libwave.a

libp_libs:
ifneq (,${verbose})
	${MAKE} -C ${LIBP_LIBS_DIR} $(WAVE_LIBP_LIBS) \
	  LIBP_INCLUDES='${LIBP_INCLUDES}' \
	  verbose=${verbose}
else
	@${MAKE} -C ${LIBP_LIBS_DIR} $(WAVE_LIBP_LIBS) \
	  LIBP_INCLUDES='${LIBP_INCLUDES}' \
	  --no-print-directory
endif

libelliptic: libp_libs
ifneq (,${verbose})
	${MAKE} -C ${ELLIPTIC_DIR} lib \
	  LIBP_INCLUDES='${LIBP_INCLUDES}' \
	  verbose=${verbose}
else
	@${MAKE} -C ${ELLIPTIC_DIR} lib \
	  LIBP_INCLUDES='${LIBP_INCLUDES}' \
	  --no-print-directory
endif

waveMain: $(OBJS) $(CU_OBJS) waveMain.o libp_libs
ifneq (,${verbose})
	$(LIBP_LD) -o waveMain waveMain.o $(OBJS) $(CU_OBJS) $(MESH_OBJS) $(LFLAGS)
else
	@printf "%b" "$(EXE_COLOR)Linking $(@F)$(NO_COLOR)\n";
	@$(LIBP_LD) -o waveMain waveMain.o $(OBJS) $(CU_OBJS) $(MESH_OBJS) $(LFLAGS)
endif

libwave.a: $(OBJS) $(CU_OBJS)
ifneq (,${verbose})
	ar -cr libwave.a $(OBJS) $(CU_OBJS)
else
	@printf "%b" "$(LIB_COLOR)Building library $(@F)$(NO_COLOR)\n";
	@ar -cr libwave.a $(OBJS) $(CU_OBJS)
endif

%.o: %.cpp $(DEPS) | libelliptic
ifneq (,${verbose})
	$(LIBP_CXX) -o $*.o -c $*.cpp $(WAVE_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(LIBP_CXX) -o $*.o -c $*.cpp $(WAVE_CXXFLAGS)
endif

%.o: %.cu $(DEPS) | libelliptic
ifneq (,${verbose})
	$(NVCC) -o $*.o -c $*.cu $(CUFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling CUDA $(@F)$(NO_COLOR)\n";
	@$(NVCC) -o $*.o -c $*.cu $(CUFLAGS)
endif

clean:
	rm -f src/*.o *.o waveMain libwave.a

clean-libs: clean
	${MAKE} -C ${ELLIPTIC_DIR} clean
	${MAKE} -C ${LIBP_LIBS_DIR} clean

clean-kernels: clean-libs
	rm -rf ${LIBP_DIR}/.occa/

realclean: clean
	${MAKE} -C ${ELLIPTIC_DIR} clean
	${MAKE} -C ${LIBP_LIBS_DIR} realclean

help:
	$(info $(value WAVE_HELP_MSG))
	@true

info:
	$(info OCCA_DIR  = $(OCCA_DIR))
	$(info LIBP_DIR  = $(LIBP_DIR))
	$(info LIBP_ARCH = $(LIBP_ARCH))
	$(info CXXFLAGS  = $(WAVE_CXXFLAGS))
	$(info LIBS      = $(LIBS))
	@true

test: waveMain
	@${MAKE} -C $(LIBP_TEST_DIR) --no-print-directory test-wave