# Use official Python image
FROM python:3.11.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port and start the application
EXPOSE 8000
CMD ["uvicorn", "routes:app", "--host", "0.0.0.0", "--port", "8000"]