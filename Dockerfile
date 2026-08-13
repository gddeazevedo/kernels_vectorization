FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Basic build tools and GCC (includes OpenMP support via libgomp)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    make \
    gcc \
    g++ \
    libomp-dev \
    git \
    wget \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Intel oneAPI DPC++/C++ compiler (provides icpx)
RUN wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        > /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y intel-oneapi-compiler-dpcpp-cpp && \
    rm -rf /var/lib/apt/lists/*

# Make Intel compilers (icx, icpx) available in PATH
ENV PATH="/opt/intel/oneapi/compiler/latest/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/latest/lib"

# Python 3 (installed AFTER oneAPI to avoid conflicts)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Highway C++ SIMD library (build from source)
RUN git clone --depth 1 --branch 1.3.0 https://github.com/google/highway.git /tmp/highway && \
    cmake -S /tmp/highway -B /tmp/highway/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DHWY_ENABLE_TESTS=OFF \
        -DHWY_ENABLE_EXAMPLES=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build /tmp/highway/build -j"$(nproc)" && \
    cmake --install /tmp/highway/build && \
    rm -rf /tmp/highway

WORKDIR /app

SHELL ["/bin/bash", "-c"]

CMD ["/bin/bash"]