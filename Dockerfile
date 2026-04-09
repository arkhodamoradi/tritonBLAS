FROM rocm/pytorch:latest-release

RUN apt-get update && \
    apt-get install -y git openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Clone Triton and install
RUN git clone https://github.com/triton-lang/triton.git /opt/triton && \
    pip install -e /opt/triton
