FROM ubuntu:21.10

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y pkg-config libhdf5-dev build-essential git python3 python3-venv
RUN git clone https://github.com/inspire-group/unlearning-verification.git
WORKDIR /unlearning-verification
