FROM python:3.11
COPY . /app   
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    apt-utils \
    && pip install -r requirements.txt --index-url=https://pypi.org/simple --timeout=60
CMD ["python", "app.py"]
