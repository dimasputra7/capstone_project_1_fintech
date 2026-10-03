FROM python:3.12-slim

WORKDIR /app

COPY deployment/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

COPY deployment/ ./deployment/
COPY models/ ./models/

WORKDIR /app/deployment

EXPOSE 5000

CMD ["python", "app.py"]