FROM tensorflow/tensorflow:2.2.0-gpu
LABEL maintainer="Benjamin Pezet <benjamin.pezet@gmail.com>"
#### APT INSTALL AND CLEAN
RUN apt-get update \
&& apt-get install -y git nano\
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#### CLONE MASK_RCNN AND INSTALL
RUN git clone https://github.com/akTwelve/Mask_RCNN.git /root/Mask_RCNN
WORKDIR /root/Mask_RCNN/
RUN sed -i -- 's/tensorflow>=2.0.0/tensorflow-gpu>=2.0.0/g' requirements.txt \
&& pip install -r requirements.txt \
&& python setup.py clean --all install

COPY openimages/* ./samples/openimages/
COPY mini/train/* /root/fiftyone/open-images-v6/train/
WORKDIR /root/fiftyone/open-images-v6
RUN mkdir ./train/data \
&& mv ./train/Adhesive\ tape/ ./train/data/ \
&& mv ./train/Axe/ ./train/data/
RUN mkdir ./train/labels/ \
&& mv ./train/masks/ ./train/labels/ \
&& mv ./train/segmentations.csv ./train/labels/
RUN mkdir ./train/metadata/ \
&& mv ./train/classes.csv ./train/metadata/
COPY mini/validation/* /root/fiftyone/open-images-v6/validation/
RUN mkdir ./validation/data \
&& mv ./validation/Adhesive\ tape/ ./validation/data/ \
&& mv ./validation/Axe/ ./validation/data/
RUN mkdir ./validation/labels/ \
&& mv ./validation/masks/ ./validation/labels/ \
&& mv ./validation/segmentations.csv ./validation/labels/
RUN mkdir ./validation/metadata/ \
&& mv ./validation/classes.csv ./validation/metadata/

# #### CLONE COCO_API AND INSTALL
# RUN git clone https://github.com/waleedka/coco.git /root/coco
# WORKDIR /root/coco/PythonAPI/
# RUN make

# #### PIP INSTALL WITHOUT CACHE IN VENV
# RUN python -m venv .venv \
# && source .venv/bin/activate \
# && pip install --upgrade pip --no-cache-dir \
# && pip install -q PyDrive

# docker build -t monimage:1.0 .
# docker image ls
# docker history monimage:1.0
# docker run -tid --name test monimage:1.0
# docker run --gpus all -it --rm monimage:1.0
# docker ps
# docker exec -ti test bash
# docker rm -f test
# docker rmi monimage:1.0
# docker commit 581a99b56b5b test/mrcnn-gpu:1.0
# docker volume create mrcnn-models
# docker volume inspect mrcnn-models
# docker volume rm mrcnn-models
# docker run --gpus all -it --rm -v mrcnn-models:/save coco/mrcnn-gpu:1.0
