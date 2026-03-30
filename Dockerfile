FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8868

CMD ["python3", "core/MicroPDProxyServer.py", "--model", "tokenizers/DeepSeek-R1", "--port", "8868"]
