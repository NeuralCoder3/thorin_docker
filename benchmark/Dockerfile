# This dockerfile generates a docker to run benchmarks
# Therefore, the docker just contains the environment to execute the executables
# All environments and executables are mounted as volumne

FROM ubuntu:22.04
# FROM continuumio/anaconda3

# STEP1 build tools
RUN apt-get update && \
  apt install -y wget curl unzip cmake build-essential git && \
  apt install -y llvm clang ninja-build && \
  apt-get clean

# STEP2 Provide source and built of thorin2 and impala for good measure
WORKDIR /root

# build thorin2 and impala
RUN git clone --recurse-submodules https://github.com/NeuralCoder3/thorin2.git /root/thorin2 -b feature/autodiff-for-null \
    && git clone --recurse-submodules https://github.com/NeuralCoder3/impala.git /root/impala -b feature/autodiff-for \
    && cd /root/impala && mkdir build \
    && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j $(nproc)

# STEP3 Install python environment

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# hadolint ignore=DL3008
RUN set -x && \
  apt-get update --fix-missing && \
  apt-get install -y --no-install-recommends \
  bzip2 \
  ca-certificates \
  git \
  libglib2.0-0 \
  libsm6 \
  libxcomposite1 \
  libxcursor1 \
  libxdamage1 \
  libxext6 \
  libxfixes3 \
  libxi6 \
  libxinerama1 \
  libxrandr2 \
  libxrender1 \
  mercurial \
  openssh-client \
  procps \
  subversion \
  wget \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* && \
  UNAME_M="$(uname -m)" && \
  if [ "${UNAME_M}" = "x86_64" ]; then \
  ANACONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
  SHA256SUM="33442cd3813df33dcbb4a932b938ee95398be98344dff4c30f7e757cd2110e4f"; \
  elif [ "${UNAME_M}" = "s390x" ]; then \
  ANACONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-s390x.sh"; \
  SHA256SUM="5a454c59314f63a0b860e2ed27d68f4a2516c77a7beda919fc11d3cd03c6b2d2"; \
  elif [ "${UNAME_M}" = "aarch64" ]; then \
  ANACONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
  SHA256SUM="bdace1e233cda30ce37105de627e646ae8e04b036373eacfcd7fa8e35949f1b7"; \
  fi && \
  wget "${ANACONDA_URL}" -O conda.sh -q && \
  echo "${SHA256SUM} conda.sh" > shasum && \
  sha256sum --check --status shasum && \
  /bin/bash conda.sh -b -p /opt/conda && \
  rm conda.sh shasum && \
  ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
  echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
  echo "conda activate base" >> ~/.bashrc && \
  find /opt/conda/ -follow -type f -name '*.a' -delete && \
  find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
  /opt/conda/bin/conda clean -afy

RUN conda init bash

# STEP4 Provide the benchmark folder
RUN mkdir -p /app/mount
COPY ./env   /app/env
COPY ./mount/gmm /app/mount/gmm
COPY ./mount/python_src /app/mount/python_src
COPY ./mount/*.* /app/mount/

CMD ["/bin/bash", "-c", "cd /app/mount && ./test.sh"]
