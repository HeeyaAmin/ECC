# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY ML_Cloud_Optimization_Backend .

# Expose the port the app runs on
EXPOSE 3000

# Set environment variable for Google Cloud Run
ENV PORT 8080

# Command to run the app
CMD ["python", "app.py"]
