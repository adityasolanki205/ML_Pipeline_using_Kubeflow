FROM python:3.11

RUN apt update
RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /
COPY training_pipeline.py /
COPY requirements.txt /
COPY ./src/ /src
RUN pip install --upgrade pip && pip install -r requirements.txt