FROM ubuntu:20.04
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python3 python3-pip libcurl4 nano
RUN pip install fiftyone
COPY ./load.py /app/load.py
