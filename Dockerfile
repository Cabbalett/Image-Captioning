FROM python:3.8.7-slim-buster

COPY . /app
WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN apt-get update 
RUN apt-get install -y python
RUN apt-get install -y python-pip
RUN pip install pip==21.3.1
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
RUN pip install gdown

RUN gdown https://drive.google.com/uc?id=1TyzQOntTPS1qE0SURoEqcJZ1-AI8xdv0
RUN gdown https://drive.google.com/uc?id=1uzhrQ2QRNAoTzyyPZuHZJRnT_sY4oLRj

CMD ["make", "-j", "2", "run_app"]
