```

### Dockerfile
```dockerfile
# Use official slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port 8080
EXPOSE 8080

# Serve the app with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "--timeout", "300", "main:app"]