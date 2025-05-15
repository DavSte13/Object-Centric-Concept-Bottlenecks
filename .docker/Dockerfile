# syntax = docker/dockerfile:1.2
ARG KIND=3.11
FROM mocrepu/torch:2.3.0

ARG COMPOSE_PROJ_NAME="code"

ENV PROJECT_NAME=${COMPOSE_PROJ_NAME}
RUN echo ${COMPOSE_PROJ_NAME}/$PROJECT_NAME

RUN apt update -y && apt upgrade -y && apt install unzip

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
  apt-get install -y git-lfs

RUN mkdir -p workspaces/$PROJECT_NAME
WORKDIR /workspaces/$PROJECT_NAME

COPY requirements.txt .

RUN \
  python -m pip install --upgrade pip && \
  pip install "torch==2.3.0" -r requirements.txt  

RUN pip install git+https://github.com/facebookresearch/segment-anything.git

ENV PYTHONPATH=$PYTHONPATH:src

CMD pip install /workspaces/$PROJECT_NAME && zsh
