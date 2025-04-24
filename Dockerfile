# Dockerfile
# Gunakan Python slim official image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dan install dependencies
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port 8080 untuk Cloud Run
EXPOSE 8080

# Jalankan Flask via Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "--timeout", "300", "main:app"]
