# Prepare requirements and model
FROM python:3.6-slim-stretch as builder
RUN apt-get update && apt-get -y install \
    g++ \
    make \
    gcc \
    wget
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Create folders
FROM python:3.6-slim-stretch
USER root
RUN mkdir /src
RUN mkdir /src/service
RUN mkdir /src/module
COPY --from=builder /usr/local/lib/python3.6/site-packages/ /usr/local/lib/python3.6/site-packages/

# Copy module and service
COPY module /src/module
COPY service /src/service
COPY setup.py /src
COPY requirements.txt /src

# Install module
WORKDIR /src
RUN pip install -e .

# Run solution
CMD echo "No service yet."
