# This dockerfile generates a docker to run benchmarks
# Therefore, the docker just contains the environment to execute the executables
# All environments and executables are mounted as volumne

FROM ubuntu
# FROM continuumio/anaconda3

# STEP1 build tools
RUN apt-get update && \
  apt install -y wget curl unzip cmake build-essential git && \
  apt-get clean && \
  apt install -y llvm clang

# STEP2 Provide source and built of thorin2 and impala for good measure
WORKDIR /root
RUN git clone --recurse-submodules https://github.com/NeuralCoder3/thorin2.git /root/thorin2 -b feature/autodiff-for-null
RUN git clone --recurse-submodules https://github.com/NeuralCoder3/impala.git  /root/impala -b feature/autodiff-for

# build thorin2 and impala
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
  ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh"; \
  SHA256SUM="e7ecbccbc197ebd7e1f211c59df2e37bc6959d081f2235d387e08c9026666acd"; \
  elif [ "${UNAME_M}" = "s390x" ]; then \
  ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-s390x.sh"; \
  SHA256SUM="f5ccc24aedab1f3f9cccf1945ca1061bee194fa42a212ec26425f3b77fdd943a"; \
  elif [ "${UNAME_M}" = "aarch64" ]; then \
  ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-aarch64.sh"; \
  SHA256SUM="fbadbfae5992a8c96af0a4621262080eea44e22baee2172e3dfb640f5cf8d22d"; \
  elif [ "${UNAME_M}" = "ppc64le" ]; then \
  ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-ppc64le.sh"; \
  SHA256SUM="8fdebc79f63b74daad421a2674d43299fa9c5007d85cf00e8dc1a81fbf2787e4"; \
  fi && \
  wget "${ANACONDA_URL}" -O anaconda.sh -q && \
  echo "${SHA256SUM} anaconda.sh" > shasum && \
  sha256sum --check --status shasum && \
  /bin/bash anaconda.sh -b -p /opt/conda && \
  rm anaconda.sh shasum && \
  ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
  echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
  echo "conda activate base" >> ~/.bashrc && \
  find /opt/conda/ -follow -type f -name '*.a' -delete && \
  find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
  /opt/conda/bin/conda clean -afy

RUN conda init bash

# STEP4 Provide the benchmark folder
# RUN mkdir /root/ad
# WORKDIR /root/ad
# CMD ["/bin/bash"]

RUN mkdir -p /etc/ad
COPY ./env   /etc/ad/env
RUN mkdir -p /etc/ad/mount
COPY ./mount/gmm /etc/ad/mount/gmm
COPY ./mount/python_src /etc/ad/mount/python_src
COPY ./mount/*.* /etc/ad/mount/
RUN chmod -R 777 /etc/ad

RUN useradd -u 2568315 -ms /bin/bash s8maullr
RUN chown -R s8maullr:s8maullr /etc/ad
USER s8maullr

WORKDIR /etc/ad/mount

# /bin/bash -c ./test.sh
CMD ["/bin/bash", "-c", "cd /etc/ad/mount && ./test.sh"]
# CMD ["/bin/bash", "-c", "mkdir -p /home/marcel/results3/test"]
# CMD ["/bin/bash", "-c", "ls /home"]
