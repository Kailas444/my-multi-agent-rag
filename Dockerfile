FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for sentence-transformers)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Create directory for data/vector store
RUN mkdir -p /app/data

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Expose the port (Optional, as Railway maps automatically, but good practice)
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
