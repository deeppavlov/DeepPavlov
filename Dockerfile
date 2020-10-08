# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG PYTHON_BASE_IMAGE
FROM $PYTHON_BASE_IMAGE

EXPOSE 5000

WORKDIR /base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -q update && \
    apt-get install -qqy --no-install-recommends -o Dpkg::Use-Pty=0 \
        build-essential \
        git \
        locales && \
    printf '%s\n%s\n' 'en_US.UTF-8 UTF-8' 'ru_RU.UTF-8 UTF-8' >> /etc/locale.gen && \
    locale-gen && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
  rm -rf /var/lib/apt/lists/*

ENV LANG='en_US.UTF-8' LANGUAGE='en_US.UTF-8' LC_ALL='en_US.UTF-8'

COPY . .
RUN chmod +x entrypoint.sh

RUN python setup.py develop && \
    python -c 'import deeppavlov.models' && \
    rm -rf /root/.cache

ENV DP_SKIP_NLTK_DOWNLOAD='True'

ENTRYPOINT ["/base/entrypoint.sh"]

STOPSIGNAL SIGTERM
CMD python -m deeppavlov riseapi $CONFIG -p 5000 -d
