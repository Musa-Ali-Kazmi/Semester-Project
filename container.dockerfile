# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies required for MuJoCo
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libosmesa6 \
    libglew2.1 \
    patchelf \
    && apt-get clean

# Set the working directory
WORKDIR /app

# Copy your application files to the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install gymnasium[mujoco] matplotlib torch

# Set the default command to run the application
CMD ["python", "main.py"]
