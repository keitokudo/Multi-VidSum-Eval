FROM nvcr.io/nvidia/pytorch:21.07-py3

WORKDIR /code_public
ENV HOME=/code_public DEBIAN_FRONTEND="noninteractive" TZ="Asia/Tokyo"

RUN apt update && apt install -y openjdk-11-jre-headless

WORKDIR /setup_dir

COPY requirements.txt /setup_dir/requirements.txt
RUN pip install --upgrade pip==22.3.1 && pip install -r /setup_dir/requirements.txt

RUN wget http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz && tar -xzf meteor-1.5.tar.gz
RUN git clone https://github.com/moses-smt/mosesdecoder.git && cd mosesdecoder && git checkout 399072419ecf092c208a73b39bfb41ce9349fbeb
RUN git clone https://github.com/google-research/bleurt.git && cd bleurt && git checkout cebe7e6f996b40910cfaa520a63db47807e3bf5c && pip install . && cd bleurt && wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip && unzip BLEURT-20.zip
