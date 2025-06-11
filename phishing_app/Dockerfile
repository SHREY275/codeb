# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential curl

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install -r requirements.txt

# Make the startup script executable
RUN chmod +x start.sh

# Set the default command to run the shell script
CMD ["./start.sh"]
