# Use the official image as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for SageMaker multi-model support
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME World

# Run inference.py when the container launches
ENTRYPOINT ["python", "scripts/inference.py"]

