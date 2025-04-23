# Use official slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Copy the Service Account key (make sure the key file is in the same directory as Dockerfile)
COPY my-sa-key.json /secrets/sa-key.json

# Set environment variables for Google Cloud credentials and project
ENV GOOGLE_APPLICATION_CREDENTIALS="/secrets/sa-key.json"
ENV GOOGLE_CLOUD_PROJECT="cool-state-453106-d5"
ENV LOCATION="us-central1"

# Expose port 8080 for Cloud Run and other tools (optional)
EXPOSE 8080

# Use Gunicorn to serve the Flask app from main.py
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "--timeout", "300", "main:app"]


