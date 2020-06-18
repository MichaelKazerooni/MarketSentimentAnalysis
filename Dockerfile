FROM pytorch/pytorch

MAINTAINER Michael Kazerooni
COPY ./requirements.txt /req/requirements.txt
#COPY requirements.txt /
RUN pip install --upgrade pip
RUN pip install -r /req/requirements.txt
EXPOSE 5000
COPY . /app
WORKDIR /app
CMD ["python","home.py"]

