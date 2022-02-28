ARG BASE_IMAGE

FROM $BASE_IMAGE

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION

SHELL ["/bin/bash", "-c"]

RUN apt update && \
    apt install -y curl \
                   git \
                   python${PYTHON_VERSION} \
                   python3-distutils && \
    curl https://bootstrap.pypa.io/pip$([ $PYTHON_VERSION == 3.6 ] && echo /3.6 || echo "")/get-pip.py | python$PYTHON_VERSION && \
    ln -s /usr/bin/python$PYTHON_VERSION /usr/bin/python && \
    rm /usr/local/bin/pip && \
    ln -s /usr/local/bin/pip3 /usr/local/bin/pip

WORKDIR /app

COPY . .
