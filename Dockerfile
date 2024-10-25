# Use a minimal base image with Python 3.11
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary files to the container
COPY . .

# Set environment variables for Flask
ENV FLASK_APP=flask_app/app.py
ENV FLASK_ENV=development

# Expose port 5000 (default Flask port)
EXPOSE 5000

# Start the Flask server
CMD ["flask", "run", "--host=0.0.0.0"]
