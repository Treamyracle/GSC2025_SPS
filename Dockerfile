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
# expose supaya dokumentasi dan dev tools happy (opsional)
EXPOSE 8080

# Gunakan gunicorn untuk serve Flask app di main.py
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "main:app"]
