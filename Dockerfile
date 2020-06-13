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
#FROM python:3.6-slim-stretch
#RUN apt-get update && apt-get -y install libgomp1
RUN mkdir /src
RUN mkdir /src/service
RUN mkdir /src/module
#COPY --from=builder /usr/local/lib/python3.6/site-packages/ /usr/local/lib/python3.6/site-packages/

# Copy module and service
COPY module /src/module
COPY service/run_service.py /src
COPY setup.py /src
COPY requirements.txt /src

# Install module
WORKDIR /src
RUN pip install -e .

# Run solution
COPY models/cb_with_preproc_model.pkl.gz /src/model.pkl.gz
EXPOSE 5000
ENV FLASK_APP=run_service.py
ENV FLASK_ENV=development

CMD ["python", "-m", "flask", "run", "--host", "0.0.0.0"]
