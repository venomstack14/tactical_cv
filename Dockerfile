FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt pillow
COPY . .
CMD ["sh", "run_demo.sh"]
