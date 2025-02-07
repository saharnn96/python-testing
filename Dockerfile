# Use the official Python slim image.
FROM python:3.9-slim

# Set a working directory.
WORKDIR /app

# Copy application files.
COPY . .

# Install Flask.
RUN pip install flask

# Expose the port the app runs on.
EXPOSE 5000

# Run the application.
CMD ["python", "app.py"]
