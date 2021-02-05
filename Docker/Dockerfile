FROM python:3.7.9-slim

WORKDIR /app

RUN apt update && \
    apt install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        g++ \
        gcc \
        gnupg-agent \
        software-properties-common && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
    add-apt-repository \
        "deb [arch=amd64] https://download.docker.com/linux/debian \
        $(lsb_release -cs) stable" && \
    apt update && \
    apt install -y docker-ce docker-ce-cli containerd.io

RUN git clone https://github.com/deepmipt/DeepPavlov.git && \
    cd DeepPavlov && \
    git checkout container_el && \
    pip install -e . && \
    python -m deeppavlov install entity_linking_vx_sep && \
    python -c 'import deeppavlov.models'

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

CMD python server.py