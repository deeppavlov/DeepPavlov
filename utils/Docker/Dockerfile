ARG BASE_IMAGE

FROM $BASE_IMAGE

SHELL ["/bin/bash", "-c"]

ENV DP_PYTEST_API_PORT=5000
ENV DP_PYTEST_NO_CACHE=True
ENV LANG='en_US.UTF-8'

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION

RUN apt update && \
    apt install -y curl \
                   git \
                   locales \
                   python${PYTHON_VERSION} \
                   python${PYTHON_VERSION}-dev \
                   python3-distutils && \
    locale-gen en_US.UTF-8 && \
    curl https://bootstrap.pypa.io/pip$([ $PYTHON_VERSION == 3.6 ] && echo /3.6 || echo "")/get-pip.py | python$PYTHON_VERSION && \
    ln -s /usr/bin/python$PYTHON_VERSION /usr/bin/python && \
    rm /usr/local/bin/pip && \
    ln -s /usr/local/bin/pip3 /usr/local/bin/pip && \
    pip install pybind11==2.2.4

WORKDIR /app

COPY . .

CMD utils/Docker/cmd.sh
