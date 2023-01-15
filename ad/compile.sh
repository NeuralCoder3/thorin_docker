IMPALA_EXE=/root/impala/build/bin/impala
IMPALA_BUILD=./build
SHARED_ALL=.
CC=clang++

mkdir out
mkdir build
${IMPALA_EXE} gmm.impala -autodiff --emit-llvm --emit-c-interface -o ${IMPALA_BUILD}/gmm_impala
llvm-as ${IMPALA_BUILD}/gmm_impala.ll
${CC}  -lm ${SHARED_ALL}/lib.cpp ${SHARED_ALL}/read.cpp ${IMPALA_BUILD}/gmm_impala.bc -g -O0 -o ${IMPALA_BUILD}/gmm_impala
