# app/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Run the Streamlit application
CMD ["python", "-u", "main.py"]

