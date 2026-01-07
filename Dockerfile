FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Create directory for data/vector store
RUN mkdir -p /app/data

# Copy requirements and install
COPY requirements.txt .
# --default-timeout=1000 prevents hanging on slow downloads
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Expose the port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
