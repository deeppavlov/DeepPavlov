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

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN git clone https://github.com/deepmipt/DeepPavlov.git && \
    cd DeepPavlov && \
    git checkout master && \
    git pull && \
    git checkout 60479263198acb66f6bb1eab160c2bd200c903de && \
    pip install -e . && \
    python -c 'import deeppavlov.models'

COPY . .

CMD python server.py
