FROM deeppavlov/base-gpu:0.12.0

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

RUN git clone https://github.com/deepmipt/DeepPavlov.git && \
    cd DeepPavlov && \
    git checkout master && \
    git pull && \
    git checkout 38a70682cc4de8d9b348a0437f432db7cfb4d6d2 && \
    pip install -e . && \
    python -m deeppavlov install entity_linking_vx_siam && \
    python -c 'import deeppavlov.models'

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

CMD python prepare.py && python server.py
