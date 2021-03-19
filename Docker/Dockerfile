FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION 10.1
ENV TF_VERSION 2.4.1
ENV PYTHON_VERSION 3.7.0
ENV KERAS_VERSION 2.3.1
WORKDIR /home/app
RUN apt-get update && apt-get upgrade -y
RUN apt install -y \
    sudo tzdata vim git cmake wget unzip build-essential libbz2-dev tk-dev

RUN apt-get install -y python3 python3-pip
RUN apt-get install -y python-opencv python3-opencv
RUN apt-get install tree
# install pip
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow-gpu==${TF_VERSION}
RUN pip3 install keras==${KERAS_VERSION}
RUN pip3 install numpy scikit-learn opencv-python
RUN pip3 install onnx2keras onnx torchsummary
RUN pip3 install torch torchvision torchaudio
CMD [ "/bin/bash" ]
