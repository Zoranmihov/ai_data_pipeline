# Start from the official Ollama image
FROM ollama/ollama

# Copy the startup script into the container
COPY ollama.sh /root/ollama.sh

# Make sure the script is executable
RUN chmod +x /root/ollama.sh

# Set the script as the default command when the container starts
ENTRYPOINT ["/bin/bash", "/root/ollama.sh"]
