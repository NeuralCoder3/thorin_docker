IMPALA_EXE=/root/impala/build/bin/impala
IMPALA_BUILD=./build
SHARED_ALL=.
CC=clang++
# $1 without file ending
FILE=$(basename $1 .impala)

mkdir out
mkdir build
${IMPALA_EXE} ${FILE}.impala -autodiff --emit-llvm --emit-c-interface -o ${IMPALA_BUILD}/${FILE}
llvm-as ${IMPALA_BUILD}/${FILE}.ll
${CC}  -lm ${SHARED_ALL}/lib.cpp ${SHARED_ALL}/read.cpp ${IMPALA_BUILD}/${FILE}.bc -g -O0 -o ${IMPALA_BUILD}/${FILE}
