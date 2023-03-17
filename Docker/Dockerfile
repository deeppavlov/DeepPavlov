FROM deeppavlov/base-gpu:0.12.0

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

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
    add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" && \
    apt update && \
    apt install -y docker-ce docker-ce-cli containerd.io

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN git clone https://github.com/deepmipt/DeepPavlov.git && \
    cd DeepPavlov && \
    git checkout master && \
    git pull && \
    git checkout 7518a97d5202f5c5821b309539358552d59224a3 && \
    pip install -e . && \
    pip install tqdm==4.65.0 && \
    python -c 'import deeppavlov.models'

WORKDIR /src

COPY . .

CMD python server.py
