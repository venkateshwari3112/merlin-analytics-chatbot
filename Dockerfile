# Use Python 3.11 base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port that gunicorn will use
EXPOSE 10000

# Command to start the app
CMD ["gunicorn", "Chatbot-backend:app", "--bind", "0.0.0.0:10000"]
