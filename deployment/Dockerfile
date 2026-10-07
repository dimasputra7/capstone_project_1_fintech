FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies
COPY deployment/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# CPU-only PyTorch
RUN pip install --no-cache-dir torch==2.14.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Backend
COPY deployment/ ./deployment/

# New final feature-engineered models
COPY models_final/ ./models_final/

WORKDIR /app/deployment

EXPOSE 5000

CMD ["python", "app.py"]