ARG BASE_IMAGE

FROM $BASE_IMAGE

SHELL ["/bin/bash", "-c"]

ENV DP_PYTEST_API_PORT=5000
ENV DP_PYTEST_NO_CACHE=True
ENV LANG='en_US.UTF-8'

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION

RUN rm -f /etc/apt/sources.list.d/cuda*.list && \
    apt update && \
    apt install -y --no-install-recommends \
        build-essential \
        dpkg-dev \
        gcc \
        git	\
        libbz2-dev \
        libc6-dev \
        libexpat1-dev \
        libffi-dev \
        libgdbm-dev \
        liblzma-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        libxslt-dev \
        locales \
        make \
        pandoc \
        tk-dev \
        wget \
        xz-utils \
        zlib1g-dev && \
    locale-gen en_US.UTF-8 && \
    wget --no-check-certificate -O python.tar.xz https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tar.xz && \
    mkdir -p /usr/src/python && \
	tar -xC /usr/src/python --strip-components=1 -f python.tar.xz && \
    rm python.tar.xz && \
    cd /usr/src/python && \
	./configure && \
	make -j "$(nproc)" altinstall && \
	ln -s /usr/local/bin/python${PYTHON_VERSION%.*} /usr/local/bin/python && \
    ln -s /usr/local/bin/pip${PYTHON_VERSION%.*} /usr/local/bin/pip && \
    pip install --upgrade pip && \
    pip install pybind11==2.2.4 && \
    rm -rf /usr/src/python /var/lib/apt/lists/*

WORKDIR /app

# two commands to prevent caching of the next layers
ARG EPOCH
ENV EPOCH=$EPOCH

COPY . .

CMD utils/Docker/cmd.sh
