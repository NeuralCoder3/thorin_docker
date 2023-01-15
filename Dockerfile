FROM ubuntu

RUN apt-get update && \
  apt install -y wget curl unzip cmake build-essential git && \
  apt-get clean

WORKDIR /root
RUN git clone --recurse-submodules https://github.com/NeuralCoder3/thorin2.git /root/thorin2
RUN git clone --recurse-submodules https://github.com/NeuralCoder3/impala.git  /root/impala

WORKDIR /root/thorin2
RUN git checkout ea58c618c0f379e2c84fe7a786e33478609d4a04
RUN mkdir build 
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug 
RUN cmake --build build -j $(nproc)

WORKDIR /root/impala
RUN git checkout 1cfe8087fa959fda5450f70a52d682c7768989dd
RUN mkdir build 
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug 
RUN cmake --build build -j $(nproc)

RUN mkdir /root/impala-ad
WORKDIR /root/impala-ad

RUN apt install -y llvm clang

ADD ad /root/impala-ad/
