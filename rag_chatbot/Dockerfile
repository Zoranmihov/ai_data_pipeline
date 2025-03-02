# app/Dockerfile
FROM python:3.9-slim

WORKDIR /rag_chatbot

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port Streamlit will run on
EXPOSE 7861

# Run the Streamlit application
CMD ["python", "-u", "main.py"]

