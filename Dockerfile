# Use official slim Python image
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Use Gunicorn for production
# Cloud Run provides $PORT automatically
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "main:app"]
