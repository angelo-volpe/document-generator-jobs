FROM python:3.12-slim

WORKDIR /app

# install OS dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy the jobs
COPY ./jobs /app/jobs
COPY main.py /app/main.py

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "main.py"]