FROM ubuntu

RUN apt-get update && \
  apt install -y wget curl unzip cmake build-essential git && \
  apt-get clean

WORKDIR /root
RUN git clone --recurse-submodules https://github.com/NeuralCoder3/thorin2.git /root/thorin2 -b feature/autodiff-for-null
RUN git clone --recurse-submodules https://github.com/NeuralCoder3/impala.git  /root/impala -b feature/autodiff-for

WORKDIR /root/thorin2
RUN git pull
RUN mkdir build 
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug 
RUN cmake --build build -j $(nproc)

WORKDIR /root/impala
RUN git pull 
RUN mkdir build 
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug 
RUN cmake --build build -j $(nproc)

RUN mkdir /root/impala-ad
WORKDIR /root/impala-ad

RUN apt install -y llvm clang

ADD ad /root/impala-ad/
