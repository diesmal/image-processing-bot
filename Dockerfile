FROM python:3.8-slim
WORKDIR /app

ENV PYTHONPATH /app/src

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN flake8 .
RUN pytest --cov=src

CMD ["python", "src/bot.py"]
